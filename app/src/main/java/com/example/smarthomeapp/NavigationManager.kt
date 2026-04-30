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
    BOTTOM_BAR
}

data class FocusPosition(
    val section: Section,
    val row: Int,
    val col: Int
)

sealed class NavigationAction {
    data class SelectDevice(val index: Int) : NavigationAction()
    data class TurnDeviceOn(val index: Int) : NavigationAction()
    data class TurnDeviceOff(val index: Int) : NavigationAction()

    data class StartRoutine(val index: Int) : NavigationAction()
    data class StopRoutine(val index: Int) : NavigationAction()

    object SelectListening : NavigationAction()
    object NavigateDevices : NavigationAction()
    object NavigateRoutines : NavigationAction()
}

class NavigationManager(
    private val columns: Int = 2,
    private val bottomBarItems: Int = 3
) {
    var deviceCount by mutableStateOf(0)
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

        val maxRow = maxDeviceRow()

        if (focused.section == Section.DEVICES) {
            focused = focused.copy(
                row = min(focused.row, maxRow),
                col = min(focused.col, lastColumnInRow(min(focused.row, maxRow)))
            )
        }
    }

    fun handle(command: VoiceCommand): NavigationAction? {
        return when (command) {
            VoiceCommand.UP -> {
                moveUp()
                null
            }

            VoiceCommand.DOWN -> {
                moveDown()
                null
            }

            VoiceCommand.LEFT -> {
                moveLeft()
                null
            }

            VoiceCommand.RIGHT -> {
                moveRight()
                null
            }

            VoiceCommand.NO -> {
                cancelSelection()
                null
            }

            VoiceCommand.YES -> {
                confirmSelection()
            }

            VoiceCommand.ON -> {
                turnOnFocused()
            }

            VoiceCommand.OFF -> {
                turnOffFocused()
            }

            VoiceCommand.GO -> {
                startFocusedRoutine()
            }

            VoiceCommand.STOP -> {
                stopFocusedRoutine()
            }
        }
    }

    private fun moveUp() {
        focused = when (focused.section) {
            Section.BOTTOM_BAR -> {
                FocusPosition(
                    Section.DEVICES,
                    maxDeviceRow(),
                    min(focused.col, lastColumnInRow(maxDeviceRow()))
                )
            }

            Section.DEVICES -> {
                if (focused.row > 0) {
                    FocusPosition(
                        Section.DEVICES,
                        focused.row - 1,
                        min(focused.col, lastColumnInRow(focused.row - 1))
                    )
                } else focused
            }
        }
    }

    private fun moveDown() {
        focused = when (focused.section) {
            Section.BOTTOM_BAR -> focused

            Section.DEVICES -> {
                val nextRow = focused.row + 1

                if (nextRow <= maxDeviceRow()) {
                    FocusPosition(
                        Section.DEVICES,
                        nextRow,
                        min(focused.col, lastColumnInRow(nextRow))
                    )
                } else {
                    FocusPosition(
                        Section.BOTTOM_BAR,
                        0,
                        min(focused.col, bottomBarItems - 1)
                    )
                }
            }
        }
    }

    private fun moveLeft() {
        val newCol = (focused.col - 1).coerceAtLeast(0)
        focused = focused.copy(col = newCol)
    }

    private fun moveRight() {
        val maxCol = when (focused.section) {
            Section.DEVICES -> lastColumnInRow(focused.row)
            Section.BOTTOM_BAR -> bottomBarItems - 1
        }

        focused = focused.copy(
            col = min(focused.col + 1, maxCol)
        )
    }

    private fun maxDeviceRow(): Int {
        if (deviceCount == 0) return 0
        return (deviceCount - 1) / columns
    }

    private fun lastColumnInRow(row: Int): Int {
        val startIndex = row * columns
        val remaining = deviceCount - startIndex

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

    fun focusedBottomBarIndex(): Int? {
        if (focused.section != Section.BOTTOM_BAR) return null
        return focused.col
    }

    private fun cancelSelection() {
        focused = FocusPosition(
            section = Section.DEVICES,
            row = 0,
            col = 0
        )
    }

    private fun confirmSelection(): NavigationAction? {
        return when (focused.section) {
            Section.DEVICES -> {
                focusedDeviceIndex()?.let {
                    NavigationAction.SelectDevice(it)
                }
            }

            Section.BOTTOM_BAR -> {
                when (focusedBottomBarIndex()) {
                    0 -> NavigationAction.NavigateDevices
                    1 -> NavigationAction.SelectListening
                    2 -> NavigationAction.NavigateRoutines
                    else -> null
                }
            }
        }
    }

    private fun turnOnFocused(): NavigationAction? {
        val index = focusedDeviceIndex() ?: return null
        return NavigationAction.TurnDeviceOn(index)
    }

    private fun turnOffFocused(): NavigationAction? {
        val index = focusedDeviceIndex() ?: return null
        return NavigationAction.TurnDeviceOff(index)
    }

    private fun startFocusedRoutine(): NavigationAction? {
        val index = focusedDeviceIndex() ?: return null
        return NavigationAction.StartRoutine(index)
    }

    private fun stopFocusedRoutine(): NavigationAction? {
        val index = focusedDeviceIndex() ?: return null
        return NavigationAction.StopRoutine(index)
    }
}