package com.example.smarthomeapp

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlin.math.min

enum class VoiceCommand {
    UP,
    DOWN,
    LEFT,
    RIGHT,
    YES,
    NO,
    ON,
    OFF,
    STOP,
    GO
}

enum class Section {
    DEVICES,
    ROUTINES,
    BOTTOM_BAR
}

data class FocusPosition(
    val section: Section,
    val row: Int,
    val col: Int
)

class NavigationManager(
    private val columns: Int = 2,
    private val bottomBarItems: Int = 3
) {

    /* ---------- APP STATE ---------- */

    var currentDestination by mutableStateOf(AppDestinations.DEVICES)
        private set

    var isListening by mutableStateOf(true)
        private set

    var devices by mutableStateOf(
        listOf(
            Device("Lights", "Living room", true, false),
            Device("Fan", "Air circulation", false, false),
            Device("TV", "Media screen", false, false),
            Device("Kitchen", "Kitchen devices", true, true),
            Device("Garage", "Garage lights", false, true),
            Device("Bedroom", "Bedroom AC", true, false)
        )
    )
        private set

    var routines by mutableStateOf(
        listOf(
            Routines("Morning Routine", "Turn on lights + coffee maker", false),
            Routines("Movie Mode", "Dim lights + TV on", false),
            Routines("Night Routine", "Turn everything off", false),
            Routines("Away Mode", "Security + save energy", false)
        )
    )
        private set

    var pendingConfirmation by mutableStateOf<PendingConfirmation?>(null)
        private set

    sealed class PendingConfirmation {
        data class TurnOnDevice(val index: Int) : PendingConfirmation()
        data class TurnOffDevice(val index: Int) : PendingConfirmation()
    }

    /* ---------- NAVIGATION STATE ---------- */

    var deviceCount by mutableStateOf(devices.size)
        private set

    var routineCount by mutableStateOf(routines.size)
        private set

    var focused by mutableStateOf(
        FocusPosition(
            section = Section.DEVICES,
            row = 0,
            col = 0
        )
    )
        private set

    fun updateDeviceCount(count: Int) {
        deviceCount = count

        val maxRow = maxGridRow()

        if (focused.section == Section.DEVICES) {
            focused = focused.copy(
                row = min(focused.row, maxRow),
                col = min(
                    focused.col,
                    lastColumnInRow(min(focused.row, maxRow))
                )
            )
        }
    }

    /* ---------- MAIN COMMAND HANDLER ---------- */

    fun handle(command: VoiceCommand) {

        if (pendingConfirmation != null) {
            when (command) {
                VoiceCommand.YES -> confirmPending()
                VoiceCommand.NO -> cancelPending()
                else -> {} // ignore everything else
            }
            return
        }

        when (command) {
            VoiceCommand.UP -> moveUp()
            VoiceCommand.DOWN -> moveDown()
            VoiceCommand.LEFT -> moveLeft()
            VoiceCommand.RIGHT -> moveRight()

            VoiceCommand.YES -> confirmSelection()
            VoiceCommand.NO -> cancelSelection()

            VoiceCommand.ON -> turnOnFocused()
            VoiceCommand.OFF -> turnOffFocused()

            VoiceCommand.GO -> startFocusedRoutine()
            VoiceCommand.STOP -> stopFocusedRoutine()
        }
    }

    /* ---------- MOVEMENT ---------- */

    private fun moveUp() {
        focused = when (focused.section) {

            Section.BOTTOM_BAR -> {
                FocusPosition(
                    section = currentSectionFromDestination(),
                    row = maxGridRow(),
                    col = min(
                        focused.col,
                        lastColumnInRow(maxGridRow())
                    )
                )
            }

            Section.DEVICES,
            Section.ROUTINES -> {
                if (focused.row > 0) {
                    FocusPosition(
                        section = focused.section,
                        row = focused.row - 1,
                        col = min(
                            focused.col,
                            lastColumnInRow(focused.row - 1)
                        )
                    )
                } else {
                    focused
                }
            }
        }
    }

    private fun moveDown() {
        focused = when (focused.section) {

            Section.BOTTOM_BAR -> focused

            Section.DEVICES,
            Section.ROUTINES -> {
                val nextRow = focused.row + 1

                if (nextRow <= maxGridRow()) {
                    FocusPosition(
                        section = focused.section,
                        row = nextRow,
                        col = min(
                            focused.col,
                            lastColumnInRow(nextRow)
                        )
                    )
                } else {
                    FocusPosition(
                        section = Section.BOTTOM_BAR,
                        row = 0,
                        col = min(
                            focused.col,
                            bottomBarItems - 1
                        )
                    )
                }
            }
        }
    }

    private fun moveLeft() {
        focused = focused.copy(
            col = (focused.col - 1).coerceAtLeast(0)
        )
    }

    private fun moveRight() {
        val maxCol = when (focused.section) {
            Section.DEVICES,
            Section.ROUTINES -> lastColumnInRow(focused.row)

            Section.BOTTOM_BAR -> bottomBarItems - 1
        }

        focused = focused.copy(
            col = min(focused.col + 1, maxCol)
        )
    }

    /* ---------- ACTIONS ---------- */

    private fun cancelSelection() {
        if (focused.section == Section.BOTTOM_BAR) {
            focused = FocusPosition(
                Section.DEVICES,
                0,
                0
            )
        }
    }

    private fun confirmSelection() {
        when (focused.section) {

            Section.BOTTOM_BAR -> {
                when (focusedBottomBarIndex()) {
                    0 -> selectDestination(AppDestinations.DEVICES)
                    1 -> isListening = !isListening
                    2 -> selectDestination(AppDestinations.ROUTINES)
                }
            }

            Section.DEVICES -> {
                // YES does nothing on devices
            }

            Section.ROUTINES -> {
                // YES does nothing on routines
            }
        }
    }

    private fun turnOnFocused() {
        if (focused.section != Section.DEVICES) return

        val index = focusedDeviceIndex() ?: return
        val device = devices[index]

        if (device.isSensible) {
            pendingConfirmation =
                PendingConfirmation.TurnOnDevice(index)
            return
        }

        turnOnDevice(index)
    }

    private fun turnOffFocused() {
        if (focused.section != Section.DEVICES) return

        val index = focusedDeviceIndex() ?: return
        val device = devices[index]

        if (device.isSensible) {
            pendingConfirmation =
                PendingConfirmation.TurnOffDevice(index)
            return
        }

        turnOffDevice(index)
    }

    fun selectDestination(destination: AppDestinations) {
        currentDestination = destination

        focused = FocusPosition(
            section = when (destination) {
                AppDestinations.DEVICES -> Section.DEVICES
                AppDestinations.ROUTINES -> Section.ROUTINES
            },
            row = 0,
            col = 0
        )
    }

    fun toggleListening() {
        isListening = !isListening
    }

    fun toggleDevice(index: Int) {
        devices = devices.mapIndexed { i, d ->
            if (i == index) d.copy(isOn = !d.isOn)
            else d
        }
    }

    fun turnOnDevice(index: Int) {
        devices = devices.mapIndexed { i, d ->
            if (i == index) d.copy(isOn = true)
            else d
        }
    }

    fun turnOffDevice(index: Int) {
        devices = devices.mapIndexed { i, d ->
            if (i == index) d.copy(isOn = false)
            else d
        }
    }

    fun toggleRoutine(index: Int) {
        routines = routines.mapIndexed { i, r ->
            if (i == index) r.copy(isOn = !r.isOn)
            else r
        }
    }

    fun startRoutine(index: Int) {
        routines = routines.mapIndexed { i, r ->
            if (i == index) r.copy(isOn = true)
            else r
        }
    }

    fun stopRoutine(index: Int) {
        routines = routines.mapIndexed { i, r ->
            if (i == index) r.copy(isOn = false)
            else r
        }
    }

    private fun startFocusedRoutine() {
        if (currentDestination != AppDestinations.ROUTINES) return

        val index = focusedRoutineIndex() ?: return
        startRoutine(index)
    }

    private fun stopFocusedRoutine() {
        if (currentDestination != AppDestinations.ROUTINES) return

        val index = focusedRoutineIndex() ?: return
        stopRoutine(index)
    }

    private fun confirmPending() {
        when (val action = pendingConfirmation) {
            is PendingConfirmation.TurnOnDevice ->
                turnOnDevice(action.index)

            is PendingConfirmation.TurnOffDevice ->
                turnOffDevice(action.index)

            null -> {}
        }

        pendingConfirmation = null
    }

    private fun cancelPending() {
        pendingConfirmation = null
    }

    /* ---------- HELPERS ---------- */

    private fun maxGridRow(): Int {
        val count = currentGridCount()
        if (count == 0) return 0
        return (count - 1) / columns
    }

    private fun lastColumnInRow(row: Int): Int {
        val count = currentGridCount()
        val startIndex = row * columns
        val remaining = count - startIndex

        return when {
            remaining <= 0 -> 0
            remaining >= columns -> columns - 1
            else -> remaining - 1
        }
    }

    fun focusedDeviceIndex(): Int? {
        if (focused.section != Section.DEVICES) return null

        val index = focused.row * columns + focused.col
        return if (index < deviceCount) index else null
    }

    fun focusedRoutineIndex(): Int? {
        if (focused.section != Section.ROUTINES) return null

        val index = focused.row * columns + focused.col
        return if (index < routineCount) index else null
    }

    fun focusedBottomBarIndex(): Int? {
        if (focused.section != Section.BOTTOM_BAR) return null
        return focused.col
    }

    private fun currentGridCount(): Int {
        return when (focused.section) {
            Section.DEVICES -> deviceCount
            Section.ROUTINES -> routineCount
            Section.BOTTOM_BAR -> 0
        }
    }

    private fun currentSectionFromDestination(): Section {
        return when (currentDestination) {
            AppDestinations.DEVICES -> Section.DEVICES
            AppDestinations.ROUTINES -> Section.ROUTINES
        }
    }

    fun isConfirming(): Boolean = pendingConfirmation != null
}