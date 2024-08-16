/*
 * Copyright (C) 2025 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
