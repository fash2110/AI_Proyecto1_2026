package com.example.smarthomeapp

import android.os.Bundle
import android.speech.tts.Voice
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.tooling.preview.Preview
import com.example.smarthomeapp.ui.theme.SmartHomeAppTheme
import androidx.compose.foundation.shape.CircleShape
import android.Manifest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import android.util.Log
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.runtime.snapshotFlow

data class Device(
    val name: String,
    val description: String,
    var isOn: Boolean,
    var isSensible: Boolean
)

data class Routines(
    val name: String,
    val description: String,
    var isOn: Boolean
)

data class PredictionResult(
    val word: String,
    val confidence: Float,
    val performed: Boolean
)
class MainActivity : ComponentActivity() {

    private lateinit var recorder: VoiceCommandRecorder
    private lateinit var navigationManager: NavigationManager

    private val requestMicPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startListening()
            } else {
                Log.w("MainActivity", "Microphone permission denied")
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        navigationManager = NavigationManager(columns = 2)
        recorder = VoiceCommandRecorder(this, navigationManager)
        recorder.loadModel()

        // Ask for permission — startListening() is called in the callback above
        requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)

        enableEdgeToEdge()
        setContent {
            SmartHomeAppTheme {
                SmartHomeAppApp(navigationManager)
            }
        }
    }

    private fun startListening() {
        lifecycleScope.launch {
            snapshotFlow { navigationManager.isListening }
                .collect { listening ->
                    if (listening) {
                        @Suppress("MissingPermission")
                        recorder.startContinuousListening()
                    } else {
                        recorder.stopContinuousListening()
                    }
                }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        recorder.release()
    }
}

enum class AppDestinations(
    val label: String,
    val icon: Int,
) {
    DEVICES("Devices", R.drawable.ic_home),
    ROUTINES("Routines", R.drawable.ic_routines)
}

@Composable
fun SmartHomeAppApp(navigationManager: NavigationManager = remember { NavigationManager(columns = 2) }) {
    Scaffold(
        bottomBar = {
            Column {
                PredictionBanner(navigationManager.lastPrediction)
                BottomBar(
                    currentDestination = navigationManager.currentDestination,
                    onNavigate = { },
                    isListening = navigationManager.isListening,
                    onToggleListening = { navigationManager.toggleListening() },
                    navigationManager = navigationManager
                )
            }
        }
    ) { innerPadding ->

        when (navigationManager.currentDestination) {

            AppDestinations.DEVICES -> DevicesScreen(
                modifier = Modifier.padding(innerPadding),
                navigationManager = navigationManager
            )

            AppDestinations.ROUTINES -> RoutinesScreen(
                modifier = Modifier.padding(innerPadding),
                navigationManager = navigationManager
            )
        }
    }
    navigationManager.pendingConfirmation?.let {
        ConfirmationDialog(navigationManager)
    }
}

@Composable
fun ListeningIndicator(
    isListening: Boolean,
    isFocused: Boolean,
    onClick: () -> Unit
) {
    val transition = rememberInfiniteTransition(label = "")

    val heights = List(3) { index ->
        transition.animateFloat(
            initialValue = 6f,
            targetValue = 24f,
            animationSpec = infiniteRepeatable(
                animation = tween(
                    durationMillis = 500,
                    delayMillis = index * 150
                ),
                repeatMode = RepeatMode.Reverse
            ),
            label = ""
        )
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clickable { onClick() }
            .border(
                width = if (isFocused) 2.dp else 0.dp,
                color = Color.Cyan,
                shape = RoundedCornerShape(12.dp)
            )
            .padding(8.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            repeat(3) { index ->
                Box(
                    modifier = Modifier
                        .width(4.dp)
                        .height(
                            if (isListening)
                                heights[index].value.dp
                            else
                                6.dp
                        )
                        .background(
                            if (isListening) Color.White else Color.Gray,
                            RoundedCornerShape(2.dp)
                        )
                )
            }
        }

        Spacer(Modifier.height(4.dp))

        Text(
            if (isListening) "listening" else "tap to listen",
            color = Color.LightGray,
            style = MaterialTheme.typography.labelSmall
        )
    }
}

@Composable
fun BottomBar(
    currentDestination: AppDestinations,
    onNavigate: (AppDestinations) -> Unit,
    isListening: Boolean,
    onToggleListening: () -> Unit,
    navigationManager: NavigationManager
) {
    val focusedBottom = navigationManager.focusedBottomBarIndex()

    NavigationBar(
        containerColor = Color(0xFF1A1F2E)
    ) {

        NavigationBarItem(
            selected = currentDestination == AppDestinations.DEVICES,
            onClick = { navigationManager.selectDestination(AppDestinations.DEVICES) },
            icon = {
                Icon(
                    painterResource(R.drawable.ic_home),
                    "Devices",
                    tint = if (focusedBottom == 0) Color.Cyan else Color.White
                )
            },
            label = { Text("Devices") }
        )

        Box(
            modifier = Modifier.weight(1f),
            contentAlignment = Alignment.Center
        ) {
            ListeningIndicator(
                isListening = isListening,
                isFocused = focusedBottom == 1,
                onClick = { navigationManager.toggleListening() }
            )
        }

        NavigationBarItem(
            selected = currentDestination == AppDestinations.ROUTINES,
            onClick = { navigationManager.selectDestination(AppDestinations.ROUTINES) },
            icon = {
                Icon(
                    painterResource(R.drawable.ic_routines),
                    "Routines",
                    tint = if (focusedBottom == 2) Color.Cyan else Color.White
                )
            },
            label = { Text("Routines") }
        )
    }
}

@Composable
fun ConfirmationDialog(
    navigationManager: NavigationManager
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.65f)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            shape = RoundedCornerShape(20.dp)
        ) {
            Column(
                Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    "Are you sure to handle this device?",
                    style = MaterialTheme.typography.headlineSmall
                )

                Spacer(Modifier.height(12.dp))

                Text("Say YES to confirm")
                Text("Say NO to cancel")
            }
        }
    }
}

@Composable
fun PredictionBanner(prediction: PredictionResult?) {
    val visible = prediction != null
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn() + slideInVertically { it },
        exit  = fadeOut() + slideOutVertically { it }
    ) {
        prediction ?: return@AnimatedVisibility
        val performed = prediction.performed
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF1A1F2E))
                .padding(horizontal = 16.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = prediction.word,
                color = Color.White,
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "${"%.0f".format(prediction.confidence * 100)}%",
                color = Color.LightGray,
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.weight(1f)
            )
            Surface(
                shape = RoundedCornerShape(4.dp),
                color = if (performed) Color(0xFF1B5E20) else Color(0xFF4A1A1A)
            ) {
                Text(
                    text = if (performed) "performed" else "ignored",
                    color = if (performed) Color(0xFF69F0AE) else Color(0xFFFF5252),
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                )
            }
        }
    }
}

@Composable
fun DevicesScreen(
    modifier: Modifier = Modifier,
    navigationManager: NavigationManager
) {
    val devices = navigationManager.devices
    val focusedIndex = navigationManager.focusedDeviceIndex()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0B1220))
            .padding(16.dp)
    ) {
        Text(
            text = "Home Dashboard",
            style = MaterialTheme.typography.headlineMedium,
            color = Color.White
        )

        Spacer(Modifier.height(12.dp))

        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            itemsIndexed(devices) { index, device ->
                DeviceCard(
                    device = device,
                    isFocused = focusedIndex == index,
                    onToggle = {
                        navigationManager.toggleDevice(index)
                    }
                )
            }
        }
    }
}

@Composable
fun DeviceCard(
    device: Device,
    isFocused: Boolean,
    onToggle: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxWidth()
    ) {

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .border(
                    width = if (isFocused) 2.dp else 0.dp,
                    color = Color.Cyan,
                    shape = RoundedCornerShape(20.dp)
                ),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(
                containerColor =
                    if (device.isOn) Color(0xFF1C2A3A)
                    else Color(0xFF151B2C)
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = device.name,
                    color = Color.White
                )

                Text(
                    text = device.description,
                    color = Color.Gray
                )

                Spacer(Modifier.height(12.dp))

                Switch(
                    checked = device.isOn,
                    onCheckedChange = { onToggle() }
                )
            }
        }

        // Warning badge overlay
        if (device.isSensible) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp),
                shape = CircleShape,
                color = Color.Red
            ) {
                Text(
                    text = "!",
                    modifier = Modifier.padding(
                        horizontal = 8.dp,
                        vertical = 2.dp
                    ),
                    color = Color.White,
                    style = MaterialTheme.typography.labelLarge
                )
            }
        }
    }
}

@Composable
fun RoutineCard(
    routine: Routines,
    isFocused: Boolean,
    onToggle: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .border(
                width = if (isFocused) 2.dp else 0.dp,
                color = Color.Cyan,
                shape = RoundedCornerShape(20.dp)
            ),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor =
                if (routine.isOn) Color(0xFF20354A)
                else Color(0xFF151B2C)
        )
    ) {
        Column(
            Modifier.padding(16.dp)
        ) {
            Text(
                routine.name,
                color = Color.White
            )

            Text(
                routine.description,
                color = Color.Gray
            )

            Spacer(Modifier.height(12.dp))

            Switch(
                checked = routine.isOn,
                onCheckedChange = {
                    onToggle()
                }
            )
        }
    }
}
@Composable
fun RoutinesScreen(
    modifier: Modifier = Modifier,
    navigationManager: NavigationManager
) {
    val routines = navigationManager.routines
    val focusedIndex = navigationManager.focusedRoutineIndex()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0B1220))
            .padding(16.dp)
    ) {
        Text(
            text = "Routines",
            style = MaterialTheme.typography.headlineMedium,
            color = Color.White
        )

        Spacer(Modifier.height(12.dp))

        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            itemsIndexed(routines) { index, routine ->
                RoutineCard(
                    routine = routine,
                    isFocused = focusedIndex == index,
                    onToggle = {
                        navigationManager.toggleRoutine(index)
                    }
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun PreviewApp() {
    SmartHomeAppTheme {
        SmartHomeAppApp()
    }
}