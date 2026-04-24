package com.example.smarthomeapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.*
import androidx.compose.material3.adaptive.navigationsuite.NavigationSuiteScaffold
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CardDefaults
import androidx.compose.ui.Alignment
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import com.example.smarthomeapp.ui.theme.SmartHomeAppTheme
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

/* ---------------- NAVIGATION ---------------- */

enum class AppDestinations(
    val label: String,
    val icon: Int,
) {
    DEVICES("Devices", R.drawable.ic_home),
    ROUTINES("Routines", R.drawable.ic_routines)
}

@Composable
fun ListeningIndicator(
    isListening: Boolean,
    onClick: () -> Unit
) {
    val infiniteTransition = rememberInfiniteTransition(label = "")

    val barHeights = List(3) { index ->
        infiniteTransition.animateFloat(
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
            .padding(8.dp)
    ) {

        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            repeat(3) { index ->
                val height = if (isListening) {
                    barHeights[index].value.dp
                } else {
                    6.dp
                }

                Box(
                    modifier = Modifier
                        .width(4.dp)
                        .height(height)
                        .background(
                            if (isListening) Color.White else Color.Gray,
                            RoundedCornerShape(2.dp)
                        )
                )
            }
        }

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = if (isListening) "listening" else "tap to listen",
            style = MaterialTheme.typography.labelSmall,
            color = Color.LightGray
        )
    }
}

@Composable
fun BottomBar(
    currentDestination: AppDestinations,
    onNavigate: (AppDestinations) -> Unit,
    isListening: Boolean,
    onToggleListening: () -> Unit
) {
    NavigationBar(containerColor = Color(0xFF1A1F2E)) {

        // Devices
        NavigationBarItem(
            selected = currentDestination == AppDestinations.DEVICES,
            onClick = { onNavigate(AppDestinations.DEVICES) },
            icon = {
                Icon(
                    painterResource(R.drawable.ic_home),
                    contentDescription = "Devices"
                )
            },
            label = { Text("Devices") }
        )

        // Indicador de escuchando
        Box(
            modifier = Modifier
                .weight(1f)
                .padding(top = 8.dp),
            contentAlignment = Alignment.Center
        ) {
            ListeningIndicator(
                isListening = isListening,
                onClick = onToggleListening
            )
        }

        // Routines
        NavigationBarItem(
            selected = currentDestination == AppDestinations.ROUTINES,
            onClick = { onNavigate(AppDestinations.ROUTINES) },
            icon = {
                Icon(
                    painterResource(R.drawable.ic_routines),
                    contentDescription = "Routines"
                )
            },
            label = { Text("Routines") }
        )
    }
}

@Composable
fun SmartHomeAppApp() {
    var currentDestination by rememberSaveable { mutableStateOf(AppDestinations.DEVICES) }
    var isListening by remember {mutableStateOf(false)}

    Scaffold(
        bottomBar = {
            BottomBar(
                currentDestination = currentDestination,
                onNavigate = { currentDestination = it },
                isListening = isListening,
                onToggleListening = {isListening = !isListening}
            )
        }
    ) { innerPadding ->
        when (currentDestination) {
            AppDestinations.DEVICES -> DevicesScreen(Modifier.padding(innerPadding))
            AppDestinations.ROUTINES -> RoutinesScreen(Modifier.padding(innerPadding))
        }
    }
}

/* ---------------- DATA MODEL ---------------- */

data class Device(
    val name: String,
    val description: String,
    var isOn: Boolean
)

/* ---------------- DEVICES SCREEN ---------------- */

@Composable
fun DevicesScreen(modifier: Modifier = Modifier) {

    var devices by remember {
        mutableStateOf(
            listOf(
                Device("Lights", "Living room & bedrooms", true),
                Device("Fan", "Air circulation", false),
                Device("TV", "Media screen", false),
                Device("Routines", "Scenes & automations", true)
            )
        )
    }

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

        Spacer(modifier = Modifier.height(8.dp))

        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(devices) { device ->

                DeviceCard(
                    device = device,
                    onToggle = {
                        devices = devices.map {
                            if (it.name == device.name) {
                                it.copy(isOn = !it.isOn)
                            } else it
                        }
                    }
                )
            }
        }
    }
}

/* ---------------- DEVICE CARD ---------------- */

@Composable
fun DeviceCard(device: Device, onToggle: () -> Unit) {

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (device.isOn)
                Color(0xFF1C2A3A)
            else
                Color(0xFF151B2C)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {

            Text(
                text = device.name,
                style = MaterialTheme.typography.titleMedium,
                color = Color.White
            )

            Text(
                text = device.description,
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )

            Spacer(modifier = Modifier.height(12.dp))

            Switch(
                checked = device.isOn,
                onCheckedChange = { onToggle() }
            )
        }
    }
}

/* ---------------- ROUTINES SCREEN ---------------- */

@Composable
fun RoutinesScreen(modifier: Modifier = Modifier) {

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

        Spacer(modifier = Modifier.height(16.dp))

        RoutineItem("Morning Routine")
        RoutineItem("Sleep Routine")
        RoutineItem("Welcome Home")
    }
}

@Composable
fun RoutineItem(name: String) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Text(
            text = name,
            modifier = Modifier.padding(16.dp)
        )
    }
}

/* ---------------- PREVIEW ---------------- */

@Preview(showBackground = true)
@Composable
fun PreviewApp() {
    SmartHomeAppTheme {
        SmartHomeAppApp()
    }
}