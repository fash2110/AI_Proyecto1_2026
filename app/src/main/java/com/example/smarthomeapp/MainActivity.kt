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
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.tooling.preview.Preview
import com.example.smarthomeapp.ui.theme.SmartHomeAppTheme
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SmartHomeAppTheme {
                SmartHomeAppApp()
            }
        }
    }
}

suspend fun runVoiceDemo(
    navigationManager: NavigationManager,
    onAction: (NavigationAction) -> Unit
) {
    val commands = listOf(
        VoiceCommand.RIGHT,
        VoiceCommand.DOWN,
        VoiceCommand.LEFT,
        VoiceCommand.DOWN,
        VoiceCommand.ON,
        VoiceCommand.UP,
        VoiceCommand.RIGHT,
        VoiceCommand.OFF,
        VoiceCommand.LEFT,
        VoiceCommand.OFF
    )

    for (command in commands) {
        val action = navigationManager.handle(command)

        if (action != null) {
            onAction(action)
        }

        delay(3000)
    }
}

enum class AppDestinations(
    val label: String,
    val icon: Int,
) {
    DEVICES("Devices", R.drawable.ic_home),
    ROUTINES("Routines", R.drawable.ic_routines)
}

data class Device(
    val name: String,
    val description: String,
    var isOn: Boolean
)

@Composable
fun SmartHomeAppApp() {
    val navigationManager = remember { NavigationManager(columns = 2) }

    var currentDestination by rememberSaveable {
        mutableStateOf(AppDestinations.DEVICES)
    }

    var isListening by remember {
        mutableStateOf(false)
    }

    var devices by remember {
        mutableStateOf(
            listOf(
                Device("Lights", "Living room & bedrooms", true),
                Device("Fan", "Air circulation", false),
                Device("TV", "Media screen", false),
                Device("Kitchen", "Kitchen devices", true),
                Device("Garage", "Garage lights", false),
                Device("Bedroom", "Bedroom AC", true)
            )
        )
    }

    LaunchedEffect(devices.size) {
        navigationManager.updateDeviceCount(devices.size)
    }

    //Demo de acciones del cursor
    LaunchedEffect(Unit) {
        runVoiceDemo(navigationManager) { action ->

            when (action) {

                is NavigationAction.TurnDeviceOn -> {
                    devices = devices.mapIndexed { index, device ->
                        if (index == action.index)
                            device.copy(isOn = true)
                        else
                            device
                    }
                }

                is NavigationAction.TurnDeviceOff -> {
                    devices = devices.mapIndexed { index, device ->
                        if (index == action.index)
                            device.copy(isOn = false)
                        else
                            device
                    }
                }

                is NavigationAction.NavigateDevices -> {
                    currentDestination = AppDestinations.DEVICES
                }

                is NavigationAction.NavigateRoutines -> {
                    currentDestination = AppDestinations.ROUTINES
                }

                is NavigationAction.SelectListening -> {
                    isListening = !isListening
                }

                is NavigationAction.SelectDevice -> {
                    devices = devices.mapIndexed { index, device ->
                        if (index == action.index)
                            device.copy(isOn = !device.isOn)
                        else
                            device
                    }
                }

                is NavigationAction.StartRoutine -> {
                    println("Start routine ${action.index}")
                }

                is NavigationAction.StopRoutine -> {
                    println("Stop routine ${action.index}")
                }
            }
        }
    }

    Scaffold(
        bottomBar = {
            BottomBar(
                currentDestination = currentDestination,
                onNavigate = { currentDestination = it },
                isListening = isListening,
                onToggleListening = { isListening = !isListening },
                navigationManager = navigationManager
            )
        }
    ) { innerPadding ->
        when (currentDestination) {
            AppDestinations.DEVICES -> DevicesScreen(
                modifier = Modifier.padding(innerPadding),
                devices = devices,
                navigationManager = navigationManager,
                onToggleDevice = { clicked ->
                    devices = devices.map {
                        if (it.name == clicked.name)
                            it.copy(isOn = !it.isOn)
                        else it
                    }
                }
            )

            AppDestinations.ROUTINES -> RoutinesScreen(
                Modifier.padding(innerPadding)
            )
        }
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
            onClick = { onNavigate(AppDestinations.DEVICES) },
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
                onClick = onToggleListening
            )
        }

        NavigationBarItem(
            selected = currentDestination == AppDestinations.ROUTINES,
            onClick = { onNavigate(AppDestinations.ROUTINES) },
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
fun DevicesScreen(
    modifier: Modifier = Modifier,
    devices: List<Device>,
    navigationManager: NavigationManager,
    onToggleDevice: (Device) -> Unit
) {
    val focusedIndex = navigationManager.focusedDeviceIndex()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0B1220))
            .padding(16.dp)
    ) {
        Text(
            "Home Dashboard",
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
                    onToggle = { onToggleDevice(device) }
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
        Column(Modifier.padding(16.dp)) {
            Text(device.name, color = Color.White)
            Text(device.description, color = Color.Gray)

            Spacer(Modifier.height(12.dp))

            Switch(
                checked = device.isOn,
                onCheckedChange = { onToggle() }
            )
        }
    }
}

@Composable
fun RoutinesScreen(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0B1220))
            .padding(16.dp)
    ) {
        Text(
            "Routines",
            style = MaterialTheme.typography.headlineMedium,
            color = Color.White
        )
    }
}

@Preview(showBackground = true)
@Composable
fun PreviewApp() {
    SmartHomeAppTheme {
        SmartHomeAppApp()
    }
}