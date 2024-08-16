package io.shubham0204.sam_android.sam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.set
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.Collections
import java.util.EnumSet

class SAMDecoder {

    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession

    // input and output node names for the decoder
    // ONNX model
    private lateinit var maskOutputName: String
    private lateinit var scoresOutputName: String

    private lateinit var imageEmbeddingInputName: String
    private lateinit var highResFeature0InputName: String
    private lateinit var highResFeature1InputName: String
    private lateinit var pointCoordinatesInputName: String
    private lateinit var pointLabelsInputName: String
    private lateinit var maskInputName: String
    private lateinit var hasMaskInputName: String

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
        val decoderInputNames = ortSession.inputNames.toList()
        val decoderOutputNames = ortSession.outputNames.toList()
        Log.i(SAMDecoder::class.simpleName, "Decoder input names: $decoderInputNames")
        Log.i(SAMDecoder::class.simpleName, "Decoder output names: $decoderOutputNames")
        imageEmbeddingInputName = decoderInputNames[0]
        highResFeature0InputName = decoderInputNames[1]
        highResFeature1InputName = decoderInputNames[2]
        pointCoordinatesInputName = decoderInputNames[3]
        pointLabelsInputName = decoderInputNames[4]
        maskInputName = decoderInputNames[5]
        hasMaskInputName = decoderInputNames[6]

        maskOutputName = decoderOutputNames[0]
        scoresOutputName = decoderOutputNames[1]
    }

    suspend fun execute(
        encoderResults: SAMEncoder.SAMEncoderResults,
        pointCoordinates: FloatBuffer,
        pointLabels: FloatBuffer,
        numLabels: Long,
        numPoints: Long,
        inputImage: Bitmap
    ): List<Bitmap> = withContext(Dispatchers.Default) {
        val imgHeight = inputImage.height
        val imgWidth = inputImage.width

        val imageEmbeddingTensor = OnnxTensor.createTensor(
            ortEnvironment,
            encoderResults.imageEmbedding,
            longArrayOf(1, 256, 64, 64),
        )
        val highResFeature0Tensor = OnnxTensor.createTensor(
            ortEnvironment,
            encoderResults.highResFeature0,
            longArrayOf(1, 32, 256, 256),
        )
        val highResFeature1Tensor = OnnxTensor.createTensor(
            ortEnvironment,
            encoderResults.highResFeature1,
            longArrayOf(1, 64, 128, 128),
        )

        val pointCoordinatesTensor = OnnxTensor.createTensor(
            ortEnvironment,
            pointCoordinates,
            longArrayOf(numLabels, numPoints, 2),
        )
        val pointLabelsTensor = OnnxTensor.createTensor(
            ortEnvironment,
            pointLabels,
            longArrayOf(numLabels, numPoints),
        )


        val maskTensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(FloatArray(numLabels.toInt() * 1 * 256 * 256) { 0f }),
            longArrayOf(numLabels, 1, 256, 256),
        )
        val hasMaskTensor = OnnxTensor.createTensor(
            ortEnvironment, FloatBuffer.wrap(floatArrayOf(0.0f)), longArrayOf(1)
        )
        val origImageSizeTensor = OnnxTensor.createTensor(
            ortEnvironment, IntBuffer.wrap(intArrayOf(imgHeight, imgWidth)), longArrayOf(2)
        )

        val outputs = ortSession.run(
            mapOf(
                imageEmbeddingInputName to imageEmbeddingTensor,
                highResFeature0InputName to highResFeature0Tensor,
                highResFeature1InputName to highResFeature1Tensor,
                pointCoordinatesInputName to pointCoordinatesTensor,
                pointLabelsInputName to pointLabelsTensor,
                maskInputName to maskTensor,
                hasMaskInputName to hasMaskTensor,
                "orig_im_size" to origImageSizeTensor
            )
        )
        val mask = (outputs[maskOutputName].get() as OnnxTensor).floatBuffer
        val scores = (outputs[scoresOutputName].get() as OnnxTensor).floatBuffer.array()
        Log.i(SAMDecoder::class.simpleName, "scores: ${scores.contentToString()}")

        // We apply masks to the input image in a parallel manner
        // by dispatching each (mask,image) pair to a new coroutine
        val bitmaps = Collections.synchronizedList(mutableListOf<Bitmap>())

        val numPredictedMasks = scores.size / numLabels.toInt()
        Log.i(SAMDecoder::class.simpleName, "Num predicted masks: $numPredictedMasks")
        Log.i(SAMDecoder::class.simpleName, "Mask size: ${mask.capacity()}")

        (0..<numLabels.toInt()).map { labelIndex ->
            launch(Dispatchers.Default) {
                // Apply mask to the input image
                // The 'on' pixels (val > 0) in the mask, will deliver an pixel
                // with alpha = 0 in the final image
                val maskStartIndex = labelIndex * numPredictedMasks * imgHeight * imgWidth
                val colorBitmap = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888)
                for (i in 0..<imgHeight) {
                    for (j in 0..<imgWidth) {
                        colorBitmap[j, i] = Color.argb(
                            if (mask[maskStartIndex + j + i * imgWidth].toInt() > 0) {
                                0
                            } else {
                                255
                            },
                            Color.red(inputImage[j, i]),
                            Color.green(inputImage[j, i]),
                            Color.blue(inputImage[j, i])
                        )
                    }
                }
                bitmaps.add(colorBitmap)
            }
        }.joinAll()

        return@withContext bitmaps
    }

    private fun saveBitmap(context: Context, image: Bitmap, name: String) {
        val fileOutputStream = FileOutputStream(File(context.filesDir.absolutePath + "/$name.png"))
        image.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
    }

}