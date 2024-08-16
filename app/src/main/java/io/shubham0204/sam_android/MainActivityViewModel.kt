package io.shubham0204.sam_android

import android.graphics.Bitmap
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel

class MainActivityViewModel : ViewModel() {

    val showBottomSheet = mutableStateOf(false)
    val selectedLabelIndex = mutableIntStateOf(0)
    val lastAddedLabel = mutableIntStateOf(0)
    val labels = mutableStateListOf("Label 0")
    val points = mutableStateListOf<MainActivity.LabelPoint>()
    val images = mutableStateListOf<Bitmap>()
    val inferenceTime = mutableIntStateOf(0)

    fun reset() {
        images.clear()
        points.clear()
        labels.clear()
        labels.add("Label 0")
        selectedLabelIndex.intValue = 0
        lastAddedLabel.intValue = 0
        showBottomSheet.value = false
        inferenceTime.intValue = 0
    }

}