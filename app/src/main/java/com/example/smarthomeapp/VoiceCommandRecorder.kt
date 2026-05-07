package com.example.smarthomeapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.annotation.RequiresPermission
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import java.nio.FloatBuffer
import kotlin.math.*

/**
 * Records a 1-second audio clip from the microphone, converts it to a
 * log mel-spectrogram, runs inference with the ONNX model, and dispatches
 * the recognised [VoiceCommand] to [NavigationManager].
 *
 * ── Preprocessing contract (matches proy1.ipynb image pipeline) ────────
 *  Sample rate   : 16 000 Hz
 *  Duration      : 1 s  → 16 000 samples  (pad end / crop end)
 *  Window        : Hann, win_length=400 (25 ms)
 *  n_fft         : 512
 *  hop_length    : 160 (10 ms)
 *  center        : true  → reflect-pad signal by n_fft/2 on both sides
 *  Mel bins      : 64
 *  f_min         : 20 Hz
 *  f_max         : 8 000 Hz  (HTK mel scale)
 *  Log           : log(mel + 1e-6)
 *  Resize        : bilinear interpolation to IMG_SIZE × IMG_SIZE (128 × 128)
 *  Normalisation : (pixel - 0.5) / 0.5   →  range [-1, 1]
 *  Tensor shape  : [1, 1, 128, 128]
 *
 * ── Label order (src/datasets/speech_commands.py AUTHORIZED_COMMANDS) ──
 *  0=yes  1=no  2=up  3=down  4=left  5=right  6=on  7=off  8=stop  9=go
 * ────────────────────────────────────────────────────────────────────────
 *
 * Usage:
 *   val recorder = VoiceCommandRecorder(context, navigationManager)
 *   recorder.loadModel()                      // call once in onCreate
 *   scope.launch { recorder.startContinuousListening() }
 *   recorder.stopContinuousListening()        // call to pause
 *   recorder.release()                        // call in onDestroy
 */
class VoiceCommandRecorder(
    private val context: Context,
    private val navigationManager: NavigationManager
) {

    /* ── constants ────────────────────────────────────────────────────── */

    companion object {
        private const val TAG = "VoiceCommandRecorder"

        // Audio
        private const val SAMPLE_RATE    = 16_000
        private const val NUM_SAMPLES    = 16_000          // 1 s at 16 kHz

        // STFT  — must match litert_parity.py exactly
        private const val N_FFT          = 512
        private const val WIN_LENGTH     = 400
        private const val HOP_LENGTH     = 160
        private const val N_MELS         = 64
        private const val F_MIN          = 20.0
        private const val F_MAX          = 8_000.0
        private const val LOG_EPS        = 1e-6f

        // Number of raw mel frames when center=true:
        //   frames = floor(num_samples / hop_length) + 1 = floor(16000/160)+1 = 101
        private const val NUM_FRAMES     = 101

        // Model was trained on 128×128 PNG images (proy1.ipynb IMG_SIZE=128).
        // The raw [64 × 101] mel is bilinear-resized to this before inference.
        private const val IMG_SIZE       = 128

        // Model
        private const val MODEL_ASSET    = "modelo_b_config3.onnx"

        // Inference gates
        private const val CONFIDENCE_THRESHOLD  = 0.70f
        private const val SILENCE_RMS_THRESHOLD = 0.01f

        // Label → command  (AUTHORIZED_COMMANDS insertion order)
        private val LABEL_TO_COMMAND = mapOf(
            0 to VoiceCommand.YES,
            1 to VoiceCommand.NO,
            2 to VoiceCommand.UP,
            3 to VoiceCommand.DOWN,
            4 to VoiceCommand.LEFT,
            5 to VoiceCommand.RIGHT,
            6 to VoiceCommand.ON,
            7 to VoiceCommand.OFF,
            8 to VoiceCommand.STOP,
            9 to VoiceCommand.GO
        )
    }

    /* ── state ────────────────────────────────────────────────────────── */

    private var ortEnv: OrtEnvironment?  = null
    private var ortSession: OrtSession?  = null

    @Volatile private var continueListening = false

    /* ── public API ───────────────────────────────────────────────────── */

    /**
     * Load the ONNX model from assets. Call once before recording.
     */
    fun loadModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val bytes = context.assets.open(MODEL_ASSET).readBytes()
            ortSession = ortEnv!!.createSession(bytes, OrtSession.SessionOptions())
            Log.i(TAG, "Model loaded. inputs=${ortSession!!.inputNames} outputs=${ortSession!!.outputNames}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX model", e)
        }
    }

    /**
     * Record one 1-second clip, run inference, dispatch the command.
     * Must be called from a coroutine — suspends during recording.
     * @return recognised [VoiceCommand] or null if silent / low-confidence.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun recordAndInfer(): VoiceCommand? = withContext(Dispatchers.IO) {
        if (!hasMicPermission()) { Log.w(TAG, "No microphone permission"); return@withContext null }
        val session = ortSession ?: run { Log.e(TAG, "Model not loaded"); return@withContext null }

        // 1. Capture
        val pcm = captureAudio() ?: return@withContext null

        // 2. Silence gate
        if (rms(pcm) < SILENCE_RMS_THRESHOLD) { Log.d(TAG, "Silence – skipped"); return@withContext null }

        // 3. Log mel spectrogram  →  FloatArray [N_MELS × NUM_FRAMES]  (64 × 101)
        val mel = computeLogMelSpectrogram(pcm)

        // 4. Bilinear resize to IMG_SIZE × IMG_SIZE  (128 × 128)
        //    Matches: transforms.Resize((IMG_SIZE, IMG_SIZE)) in the notebook
        val resized = bilinearResize(mel, N_MELS, NUM_FRAMES, IMG_SIZE, IMG_SIZE)

        // 5. Min-max scale to [0, 1]  →  matches transforms.ToTensor() on a
        //    grayscale PNG whose pixels were saved from a normalised float image
        val minVal = resized.min()
        val maxVal = resized.max()
        val range  = (maxVal - minVal).coerceAtLeast(LOG_EPS)
        for (i in resized.indices) resized[i] = (resized[i] - minVal) / range

        // 6. Normalize: (x - 0.5) / 0.5  →  range [-1, 1]
        //    Matches: transforms.Normalize(mean=[0.5], std=[0.5]) in the notebook
        for (i in resized.indices) resized[i] = (resized[i] - 0.5f) / 0.5f

        // 7. Build ONNX tensor  shape [1, 1, IMG_SIZE, IMG_SIZE]
        val shape  = longArrayOf(1L, 1L, IMG_SIZE.toLong(), IMG_SIZE.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv!!, FloatBuffer.wrap(resized), shape)

        // 8. Inference
        val inputName = session.inputNames.iterator().next()
        val results   = session.run(mapOf(inputName to tensor))

        // 9. Parse logits  [1, num_classes]
        val raw = results[0].value
        val logits: FloatArray = when (raw) {
            is Array<*>   -> (raw[0] as FloatArray)
            is FloatArray -> raw
            else -> { Log.e(TAG, "Unexpected output type: ${raw?.javaClass}"); return@withContext null }
        }

        // 10. Softmax → best class
        val probs      = softmax(logits)
        val bestIdx    = probs.indices.maxByOrNull { probs[it] } ?: return@withContext null
        val confidence = probs[bestIdx]
        Log.d(TAG, "class=$bestIdx confidence=${"%.3f".format(confidence)}")

        val word = LABEL_TO_COMMAND[bestIdx]?.name ?: "UNKNOWN"

        if (confidence < CONFIDENCE_THRESHOLD) {
            Log.d(TAG, "Low confidence – ignored")
            withContext(Dispatchers.Main) {
                navigationManager.updatePrediction(PredictionResult(word, confidence, performed = false))
            }
            return@withContext null
        }

        val command = LABEL_TO_COMMAND[bestIdx] ?: run { Log.w(TAG, "Unknown class $bestIdx"); return@withContext null }

        // 11. Dispatch on main thread
        withContext(Dispatchers.Main) {
            if (navigationManager.isListening) {
                Log.i(TAG, "Dispatching: $command")
                navigationManager.updatePrediction(PredictionResult(word, confidence, performed = true))
                navigationManager.handle(command)
            } else {
                navigationManager.updatePrediction(PredictionResult(word, confidence, performed = false))
            }
        }
        command
    }

    /**
     * Continuous record-infer loop. Launch in a lifecycle-bound coroutine scope.
     * Each iteration captures one non-overlapping 1-second window.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    suspend fun startContinuousListening() {
        continueListening = true
        Log.i(TAG, "Continuous listening started")
        while (continueListening) {
            recordAndInfer()
        }
        Log.i(TAG, "Continuous listening stopped")
    }

    /** Stop the loop after the current window finishes. */
    fun stopContinuousListening() { continueListening = false }

    /** Release ONNX resources. Call in onDestroy. */
    fun release() {
        stopContinuousListening()
        ortSession?.close(); ortSession = null
        ortEnv?.close();     ortEnv     = null
        Log.i(TAG, "Released")
    }

    /* ── audio capture ────────────────────────────────────────────────── */

    /**
     * Record [NUM_SAMPLES] PCM-16 samples and return float32 in [-1, 1].
     * Short recordings are zero-padded (pad_end policy).
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun captureAudio(): FloatArray? {
        val minBuf  = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val bufSize = maxOf(minBuf, NUM_SAMPLES * Short.SIZE_BYTES)

        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize
        )
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord init failed"); rec.release(); return null
        }

        val raw  = ShortArray(NUM_SAMPLES)
        rec.startRecording()
        val read = rec.read(raw, 0, NUM_SAMPLES)
        rec.stop(); rec.release()

        if (read < NUM_SAMPLES) Log.w(TAG, "Short read: $read / $NUM_SAMPLES (zero-padded)")

        return FloatArray(NUM_SAMPLES) { i -> raw.getOrElse(i) { 0 }.toFloat() / 32768.0f }
    }

    /* ── spectrogram ──────────────────────────────────────────────────── */

    /**
     * Compute log mel-spectrogram matching [litert_parity.py]:
     *
     *  1. Reflect-pad by n_fft/2 on both sides (center=true).
     *  2. Slice NUM_FRAMES Hann-windowed frames (win_length=400, zero-pad to n_fft=512).
     *  3. Power spectrum via in-place FFT.
     *  4. HTK mel filter bank (n_mels=64, 20–8000 Hz).
     *  5. log(energy + 1e-6).
     *
     * Returns flat FloatArray [N_MELS × NUM_FRAMES], row-major.
     */
    private fun computeLogMelSpectrogram(pcm: FloatArray): FloatArray {
        val pad    = N_FFT / 2                              // 256
        val padded = FloatArray(NUM_SAMPLES + 2 * pad)

        // Reflect-pad left  (pcm[pad-1], pcm[pad-2], …, pcm[0])
        for (i in 0 until pad) padded[i] = pcm[(pad - i).coerceAtMost(NUM_SAMPLES - 1)]
        // Copy signal
        pcm.copyInto(padded, pad)
        // Reflect-pad right  (pcm[N-2], pcm[N-3], …)
        for (i in 0 until pad) {
            padded[pad + NUM_SAMPLES + i] = pcm[(NUM_SAMPLES - 2 - i).coerceAtLeast(0)]
        }

        val hann    = hannWindow(WIN_LENGTH)
        val fftSize = N_FFT                                 // 512 is already a power of 2
        val fftBins = N_FFT / 2 + 1                         // 257

        // Build HTK mel filter bank once
        val melFB = buildHtkMelFilterBank(N_MELS, fftBins, SAMPLE_RATE, F_MIN, F_MAX)

        val out = FloatArray(N_MELS * NUM_FRAMES)

        val fftBuf = DoubleArray(fftSize * 2)               // reused each frame

        for (t in 0 until NUM_FRAMES) {
            val start = t * HOP_LENGTH

            // Zero the buffer
            fftBuf.fill(0.0)

            // Hann-windowed frame (WIN_LENGTH samples), zero-padded to N_FFT
            for (i in 0 until WIN_LENGTH) {
                val s = start + i
                fftBuf[i * 2] = if (s < padded.size) (padded[s] * hann[i]).toDouble() else 0.0
            }

            fft(fftBuf, fftSize)

            // Power spectrum + mel filters + log
            for (m in 0 until N_MELS) {
                val filter = melFB[m]
                var energy = 0.0
                for (k in 0 until fftBins) {
                    val re = fftBuf[k * 2]
                    val im = fftBuf[k * 2 + 1]
                    energy += filter[k] * (re * re + im * im)
                }
                out[m * NUM_FRAMES + t] = ln(energy.toFloat().coerceAtLeast(LOG_EPS))
            }
        }
        return out
    }

    /**
     * Bilinear resize of a [srcH × srcW] float image to [dstH × dstW].
     * Replicates torchvision transforms.Resize bilinear mode.
     * Input/output are flat row-major FloatArrays.
     */
    private fun bilinearResize(
        src: FloatArray, srcH: Int, srcW: Int, dstH: Int, dstW: Int
    ): FloatArray {
        val dst = FloatArray(dstH * dstW)
        val scaleY = srcH.toFloat() / dstH
        val scaleX = srcW.toFloat() / dstW
        for (y in 0 until dstH) {
            val srcY = (y + 0.5f) * scaleY - 0.5f
            val y0 = srcY.toInt().coerceIn(0, srcH - 1)
            val y1 = (y0 + 1).coerceIn(0, srcH - 1)
            val fy = (srcY - y0).coerceIn(0f, 1f)
            for (x in 0 until dstW) {
                val srcX = (x + 0.5f) * scaleX - 0.5f
                val x0 = srcX.toInt().coerceIn(0, srcW - 1)
                val x1 = (x0 + 1).coerceIn(0, srcW - 1)
                val fx = (srcX - x0).coerceIn(0f, 1f)
                val v = src[y0 * srcW + x0] * (1 - fy) * (1 - fx) +
                        src[y0 * srcW + x1] * (1 - fy) * fx +
                        src[y1 * srcW + x0] * fy * (1 - fx) +
                        src[y1 * srcW + x1] * fy * fx
                dst[y * dstW + x] = v
            }
        }
        return dst
    }

    private fun rms(x: FloatArray) =
        sqrt(x.fold(0.0) { acc, v -> acc + v * v } / x.size).toFloat()

    private fun softmax(logits: FloatArray): FloatArray {
        val max  = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum  = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    /* ── DSP helpers ──────────────────────────────────────────────────── */

    private fun hannWindow(size: Int) =
        FloatArray(size) { n -> (0.5 * (1.0 - cos(2.0 * PI * n / (size - 1)))).toFloat() }

    /**
     * In-place Cooley-Tukey radix-2 DIT FFT.
     * [data] = interleaved real/imag doubles, length = 2 * n  (n must be power of 2).
     */
    private fun fft(data: DoubleArray, n: Int) {
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                var t = data[2*i];     data[2*i]     = data[2*j];     data[2*j]     = t
                t = data[2*i + 1]; data[2*i + 1] = data[2*j + 1]; data[2*j + 1] = t
            }
        }
        var len = 2
        while (len <= n) {
            val half = len / 2
            val wr = cos(-2.0 * PI / len)
            val wi = sin(-2.0 * PI / len)
            var i = 0
            while (i < n) {
                var ur = 1.0; var ui = 0.0
                for (k in 0 until half) {
                    val er = data[2*(i+k+half)]   * ur - data[2*(i+k+half)+1] * ui
                    val ei = data[2*(i+k+half)]   * ui + data[2*(i+k+half)+1] * ur
                    data[2*(i+k+half)]   = data[2*(i+k)]   - er
                    data[2*(i+k+half)+1] = data[2*(i+k)+1] - ei
                    data[2*(i+k)]       += er
                    data[2*(i+k)+1]     += ei
                    val nr = ur * wr - ui * wi; ui = ur * wi + ui * wr; ur = nr
                }
                i += len
            }
            len = len shl 1
        }
    }

    /**
     * HTK-scale triangular mel filter bank.
     * Bin mapping:  bin = floor((n_fft + 1) * hz / sample_rate)
     * — identical to the formula in src/preprocess/mel.py.
     */
    private fun buildHtkMelFilterBank(
        nMels: Int, fftBins: Int, sampleRate: Int, fMin: Double, fMax: Double
    ): Array<FloatArray> {
        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        fun melToHz(mel: Double) = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

        val mMin = hzToMel(fMin)
        val mMax = hzToMel(fMax)

        val melPts = DoubleArray(nMels + 2) { i -> melToHz(mMin + i * (mMax - mMin) / (nMels + 1)) }

        // floor((n_fft + 1) * hz / sr)  — exact match to mel.py
        val bins = IntArray(nMels + 2) { i ->
            floor((N_FFT + 1) * melPts[i] / sampleRate).toInt().coerceIn(0, fftBins - 1)
        }

        return Array(nMels) { m ->
            FloatArray(fftBins) { k ->
                when {
                    k > bins[m] && k < bins[m + 1] ->
                        (k - bins[m]).toFloat() / (bins[m + 1] - bins[m]).toFloat().coerceAtLeast(1f)
                    k >= bins[m + 1] && k <= bins[m + 2] ->
                        (bins[m + 2] - k).toFloat() / (bins[m + 2] - bins[m + 1]).toFloat().coerceAtLeast(1f)
                    else -> 0f
                }
            }
        }
    }

    /* ── permission ───────────────────────────────────────────────────── */

    private fun hasMicPermission() =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
}