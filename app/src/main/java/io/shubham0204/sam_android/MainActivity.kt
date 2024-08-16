package io.shubham0204.sam_android

import AppProgressDialog
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.PointF
import androidx.exifinterface.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Tag
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.toSize
import androidx.lifecycle.viewmodel.compose.viewModel
import hideProgressDialog
import io.shubham0204.sam_android.sam.SAMDecoder
import io.shubham0204.sam_android.sam.SAMEncoder
import io.shubham0204.sam_android.ui.components.AppAlertDialog
import io.shubham0204.sam_android.ui.components.createAlertDialog
import io.shubham0204.sam_android.ui.theme.SAMAndroidTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import setProgressDialogText
import showProgressDialog
import java.nio.FloatBuffer
import kotlin.time.DurationUnit
import kotlin.time.measureTimedValue


class MainActivity : ComponentActivity() {

    private val encoder = SAMEncoder()
    private val decoder = SAMDecoder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            SAMAndroidTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .verticalScroll(rememberScrollState())
                            .padding(innerPadding)
                    ) {

                        val viewModel = viewModel<MainActivityViewModel>()

                        var image by remember { mutableStateOf<Bitmap?>(null) }
                        val outputImages = remember { viewModel.images }
                        val points = remember { viewModel.points }
                        var isReady by remember { mutableStateOf(false) }
                        var viewPortDims by remember { mutableStateOf<Size?>(null) }

                        LaunchedEffect(0) {
                            try {
                                showProgressDialog()
                                setProgressDialogText("Loading models...")
                                encoder.init(
                                    "/data/local/tmp/sam/encoder_base_plu.onnx"
                                )
                                decoder.init(
                                    "/data/local/tmp/sam/decoder_base_plus_float32.onnx"
                                )
                                isReady = true
                                hideProgressDialog()
                            } catch (e: Exception) {
                                hideProgressDialog()
                                createAlertDialog(
                                    dialogTitle = "Error",
                                    dialogText = "An error occurred: ${e.message}",
                                    dialogPositiveButtonText = "Close",
                                    dialogNegativeButtonText = null,
                                    onPositiveButtonClick = { finish() },
                                    onNegativeButtonClick = null
                                )
                            }

                        }

                        val pickMediaLauncher = rememberLauncherForActivityResult(
                            contract = ActivityResultContracts.PickVisualMedia()
                        ) {
                            if (it != null) {
                                val bitmap = getFixedBitmap(it)
                                image = bitmap
                                viewModel.reset()
                            }
                        }

                        Row(
                            modifier = Modifier
                                .padding(horizontal = 8.dp)
                                .fillMaxWidth()
                        ) {

                            Button(modifier = Modifier
                                .fillMaxWidth()
                                .padding(4.dp)
                                .weight(1f),
                                onClick = {
                                    viewModel.showBottomSheet.value = true
                                }) {
                                Icon(
                                    imageVector = Icons.Default.Tag,
                                    contentDescription = "Choose Label For Points"
                                )
                                Text(text = "Choose Label For Points")
                            }
                        }

                        Row(
                            modifier = Modifier
                                .padding(horizontal = 8.dp)
                                .fillMaxWidth()
                        ) {
                            Button(modifier = Modifier
                                .fillMaxWidth()
                                .padding(4.dp)
                                .weight(1f),
                                enabled = isReady && (image != null),
                                onClick = {
                                    image?.let { bitmap ->
                                        processInputPoints(
                                            bitmap, points, viewPortDims, viewModel
                                        )
                                    }
                                }) {
                                Icon(
                                    imageVector = Icons.Default.Layers,
                                    contentDescription = "Segment"
                                )
                                Text(text = "Segment!")
                            }
                            Button(modifier = Modifier
                                .fillMaxWidth()
                                .padding(4.dp)
                                .weight(1f),
                                enabled = isReady,
                                onClick = {
                                    pickMediaLauncher.launch(
                                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                    )
                                }) {
                                Icon(
                                    imageVector = Icons.Default.Image,
                                    contentDescription = "Select Image"
                                )
                                Text(text = "Select Image")
                            }
                        }

                        if (image != null) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Currently selected label: Label ${viewModel.selectedLabelIndex.intValue}",
                                fontSize = 12.sp,
                                color = Color.DarkGray,
                                textAlign = TextAlign.Center,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(4.dp)
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Box {
                                Image(bitmap = image!!.asImageBitmap(),
                                    contentScale = ContentScale.Fit,
                                    contentDescription = "Selected Image",
                                    modifier = Modifier
                                        .pointerInput(Unit) {
                                            detectTapGestures(onLongPress = {
                                                val newPoints =
                                                    points.filter { it.label != viewModel.selectedLabelIndex.intValue }
                                                points.clear()
                                                points.addAll(newPoints)
                                                Toast
                                                    .makeText(
                                                        this@MainActivity,
                                                        "All guide-points removed",
                                                        Toast.LENGTH_LONG
                                                    )
                                                    .show()
                                            }, onTap = { offset ->
                                                points.add(
                                                    LabelPoint(
                                                        viewModel.selectedLabelIndex.intValue,
                                                        PointF(offset.x, offset.y)
                                                    )
                                                )
                                            })
                                        }
                                        .onGloballyPositioned {
                                            viewPortDims = it.size.toSize()
                                        })
                                Spacer(modifier = Modifier
                                    .fillMaxSize()
                                    .drawWithCache {
                                        onDrawBehind {
                                            points
                                                .filter { labelPoint -> labelPoint.label == viewModel.selectedLabelIndex.intValue }
                                                .forEach { labelPoint ->
                                                    drawCircle(
                                                        color = Color.Black,
                                                        radius = 15f,
                                                        center = Offset(
                                                            labelPoint.point.x, labelPoint.point.y
                                                        )
                                                    )
                                                    drawCircle(
                                                        color = Color.Yellow,
                                                        radius = 12f,
                                                        center = Offset(
                                                            labelPoint.point.x, labelPoint.point.y
                                                        )
                                                    )
                                                }
                                        }
                                    })
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Tap on the image to insert a guide-point\nLong-press to remove all guide-points for the current label",
                                fontSize = 12.sp,
                                color = Color.DarkGray,
                                textAlign = TextAlign.Center,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp)
                            )
                        }
                        if (outputImages.isNotEmpty()) {
                            Text(
                                modifier = Modifier.padding(4.dp),
                                fontSize = 18.sp,
                                text = "Segmented Images (${viewModel.inferenceTime.intValue} s)"
                            )
                        }
                        outputImages.forEach {
                            Image(
                                modifier = Modifier.background(Color.Black.copy(green = 1.0f)),
                                bitmap = it.asImageBitmap(),
                                contentDescription = "Segmented image"
                            )
                        }

                        AppAlertDialog()
                        AppProgressDialog()
                        ManageLabelsBottomSheet(viewModel)
                    }
                }
            }
        }

    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun ManageLabelsBottomSheet(viewModel: MainActivityViewModel) {
        val sheetState = rememberModalBottomSheetState()
        val scope = rememberCoroutineScope()
        var showBottomSheet by remember { viewModel.showBottomSheet }
        val labels = remember { viewModel.labels }
        var lastAddedLabel by remember { viewModel.lastAddedLabel }
        var selectedLabelIndex by remember { viewModel.selectedLabelIndex }

        if (showBottomSheet) {
            ModalBottomSheet(
                containerColor = Color.White,
                onDismissRequest = { showBottomSheet = false },
                sheetState = sheetState
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 16.dp)
                ) {
                    Row {
                        Text(
                            text = "Manage Labels",
                            fontSize = 18.sp,
                            modifier = Modifier
                                .padding(8.dp)
                                .weight(1f)
                        )
                        IconButton(onClick = {
                            scope.launch { sheetState.hide() }.invokeOnCompletion {
                                if (!sheetState.isVisible) {
                                    showBottomSheet = false
                                }
                            }
                        }) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Close Panel"
                            )
                        }
                    }
                    LazyColumn {
                        itemsIndexed(labels) { index, item ->
                            Text(text = item, modifier = Modifier
                                .clickable {
                                    selectedLabelIndex = index
                                }
                                .background(if (selectedLabelIndex == index) Color.Cyan else Color.White)
                                .padding(8.dp)
                                .fillMaxWidth())
                        }
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    Row {
                        Button(
                            modifier = Modifier
                                .padding(4.dp)
                                .fillMaxWidth()
                                .weight(1f),
                            onClick = {
                                lastAddedLabel += 1
                                labels.add("Label $lastAddedLabel")
                            }) {
                            Text(text = "Add Label")
                        }
                        Button(
                            modifier = Modifier
                                .padding(4.dp)
                                .fillMaxWidth()
                                .weight(1f),
                            onClick = {
                                labels.removeAt(selectedLabelIndex)
                            }) {
                            Text(text = "Remove Label")
                        }
                    }
                }

            }
        }
    }

    private fun processInputPoints(
        bitmap: Bitmap,
        points: List<LabelPoint>,
        viewPortDims: Size?,
        viewModel: MainActivityViewModel
    ) {
        CoroutineScope(Dispatchers.Default).launch {
            try {
                showProgressDialog()
                setProgressDialogText("Performing image segmentation...")
                val pointsGroupByLabel = points.groupBy { it.label }
                val maxPoints = pointsGroupByLabel.maxOfOrNull { it.value.size } ?: return@launch
                val labelsCount = pointsGroupByLabel.keys.size

                val labelsBuffer = FloatBuffer.allocate(1 * labelsCount * maxPoints)
                val pointsBuffer = FloatBuffer.allocate(1 * labelsCount * maxPoints * 2)

                for ((label, labelPoints) in pointsGroupByLabel) {
                    labelPoints.forEach {
                        pointsBuffer.put((it.point.x / viewPortDims?.width!!) * 1024f)
                        pointsBuffer.put((it.point.y / viewPortDims.height) * 1024f)
                    }
                    repeat(maxPoints - labelPoints.size) {
                        pointsBuffer.put(0f)
                        pointsBuffer.put(0f)
                    }
                    repeat(labelPoints.size) {
                        labelsBuffer.put((label + 1).toFloat())
                    }
                    repeat(maxPoints - labelPoints.size) {
                        labelsBuffer.put(-1f)
                    }
                }
                pointsBuffer.rewind()
                labelsBuffer.rewind()

                val (imagesWithMask, time) = measureTimedValue {
                    decoder.execute(
                        encoder.execute(bitmap),
                        pointsBuffer,
                        labelsBuffer,
                        labelsCount.toLong(),
                        maxPoints.toLong(),
                        bitmap
                    )
                }
                withContext(Dispatchers.Main) {
                    viewModel.inferenceTime.intValue = time.toInt(DurationUnit.SECONDS)
                    hideProgressDialog()
                    viewModel.images.clear()
                    viewModel.images.addAll(imagesWithMask)
                }
            } catch (e: Exception) {
                hideProgressDialog()
                createAlertDialog(
                    dialogTitle = "Error",
                    dialogText = "An error occurred: ${e.message}",
                    dialogPositiveButtonText = "Close",
                    dialogNegativeButtonText = null,
                    onPositiveButtonClick = { finish() },
                    onNegativeButtonClick = null
                )
            }
        }
    }

    private fun getFixedBitmap(imageFileUri: Uri): Bitmap {
        var imageBitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(imageFileUri))
        val exifInterface = ExifInterface(contentResolver.openInputStream(imageFileUri)!!)
        imageBitmap = when (exifInterface.getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED
        )) {
            ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(imageBitmap, 90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(imageBitmap, 180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(imageBitmap, 270f)
            else -> imageBitmap
        }
        return imageBitmap
    }

    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, false)
    }

    data class LabelPoint(
        val label: Int, val point: PointF
    )

}