package io.shubham0204.sam_android.sam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.get
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.util.EnumSet

class SAMEncoder {

    data class SAMEncoderResults(
        val imageEmbedding: FloatBuffer,
        val highResFeature0: FloatBuffer,
        val highResFeature1: FloatBuffer
    )

    private val inputDim = 1024
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var inputName: String
    private lateinit var imageEmbeddingOutputName: String
    private lateinit var highResFeature0OutputName: String
    private lateinit var highResFeature1OutputName: String

    private val mean = floatArrayOf(
        0.485f, 0.456f, 0.406f
    )
    private val std = floatArrayOf(
        0.229f, 0.224f, 0.225f
    )

    suspend fun init(
        modelPath: String, useFP16: Boolean = false, useXNNPack: Boolean = false
    ) = withContext(Dispatchers.IO) {
        ortEnvironment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            if (useFP16) {
                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            if (useXNNPack) {
                addXnnpack(
                    mapOf(
                        "intra_op_num_threads" to "2"
                    )
                )
            }
        }
        ortSession = ortEnvironment.createSession(modelPath, options)
        inputName = ortSession.inputNames.first()
        val outputNames = ortSession.outputNames.toList()
        Log.i(SAMEncoder::class.simpleName, "Encoder input names: $inputName")
        Log.i(SAMEncoder::class.simpleName, "Encoder output names: $outputNames")
        highResFeature0OutputName = outputNames[0]
        highResFeature1OutputName = outputNames[1]
        imageEmbeddingOutputName = outputNames[2]
    }

    suspend fun execute(inputImage: Bitmap) = withContext(Dispatchers.IO) {
        // Resize the image to the model's required input size
        val resizedImage = Bitmap.createScaledBitmap(
            inputImage, inputDim, inputDim, true
        )

        // Create a FloatBuffer to store the normalized image pixels
        // The model requires the image in the shape (1, C, H, W)
        val imagePixels = FloatBuffer.allocate(1 * resizedImage.width * resizedImage.height * 3)
        imagePixels.rewind()
        for (i in 0 until resizedImage.height) {
            for (j in 0 until resizedImage.width) {
                imagePixels.put(
                    ((Color.red(resizedImage[j, i]).toFloat() / 255.0f) - mean[0]) / std[0]
                )
            }
        }
        for (i in 0 until resizedImage.height) {
            for (j in 0 until resizedImage.width) {
                imagePixels.put(
                    ((Color.blue(resizedImage[j, i]).toFloat() / 255.0f) - mean[1]) / std[1]
                )
            }
        }
        for (i in 0 until resizedImage.height) {
            for (j in 0 until resizedImage.width) {
                imagePixels.put(
                    ((Color.green(resizedImage[j, i]).toFloat() / 255.0f) - mean[2]) / std[2]
                )
            }
        }
        imagePixels.rewind()

        // Perform inference, and return the output tensors
        val imageTensor = OnnxTensor.createTensor(
            ortEnvironment,
            imagePixels,
            longArrayOf(1, 3, inputDim.toLong(), inputDim.toLong()),
        )
        val outputs = ortSession.run(mapOf(inputName to imageTensor))
        val highResFeature0 = outputs[highResFeature0OutputName].get() as OnnxTensor
        val highResFeature1 = outputs[highResFeature1OutputName].get() as OnnxTensor
        val imageEmbedding = outputs[imageEmbeddingOutputName].get() as OnnxTensor
        return@withContext SAMEncoderResults(
            imageEmbedding.floatBuffer, highResFeature0.floatBuffer, highResFeature1.floatBuffer
        )
    }
}