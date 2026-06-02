package com.shotcrete.myanmartts

import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.io.FileOutputStream
import java.util.HashMap
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // vocab.json အတိုင်း စာလုံးတစ်ခုချင်းစီရဲ့ ID အစစ်အမှန်များ
    private val vocabMap = mapOf(
        '်' to 0L, 'ာ' to 1L, 'ု' to 2L, 'ိ' to 3L, 'း' to 4L, 'ေ' to 5L, 'သ' to 6L, 'က' to 7L,
        'င' to 8L, 'တ' to 9L, '့' to 10L, 'မ' to 11L, 'ြ' to 12L, 'ည' to 13L, 'ရ' to 14L, 'အ' to 15L,
        'န' to 16L, 'လ' to 17L, 'ှ' to 18L, 'ပ' to 19L, 'စ' to 20L, 'ခ' to 21L, 'ျ' to 22L, 'ူ' to 23L,
        'ွ' to 24L, 'ါ' to 25L, 'ထ' to 26L, 'ဖ' to 27L, 'ံ' to 28L, 'ယ' to 29L, 'ဆ' to 30L, 'ီ' to 31L,
        'ဲ' to 32L, 'ဟ' to 33L, 'ဘ' to 34L, 'ဝ' to 35L, '္' to 36L, 'ဉ' to 37L, 'ဤ' to 38L, 'ဇ' to 39L,
        'ဒ' to 40L, 'ဂ' to 41L, 'ဦ' to 42L, 'ဏ' to 43L, 'ဗ' to 44L, 'ဓ' to 45L, 'ဧ' to 46L, 'ဥ' to 47L,
        'ဩ' to 48L, 'ဌ' to 49L, 'ဋ' to 50L, '\'' to 51L, 'ဣ' to 52L, 'ဍ' to 53L, 'ဿ' to 54L, 'ဈ' to 55L,
        ' ' to 56L
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val inputText = findViewById<EditText>(R.id.inputText)
        val speakButton = findViewById<Button>(R.id.speakButton)

        // ၁။ Engine စတင်နှိုးခြင်း
        thread(start = true) {
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                val modelFile = File(cacheDir, "model.onnx")
                if (!modelFile.exists()) {
                    assets.open("model.onnx").use { inputStream ->
                        FileOutputStream(modelFile).use { outputStream ->
                            val buffer = ByteArray(4 * 1024)
                            var read: Int
                            while (inputStream.read(buffer).also { read = it } != -1) {
                                outputStream.write(buffer, 0, read)
                            }
                            outputStream.flush()
                        }
                    }
                }
                ortSession = ortEnv?.createSession(modelFile.absolutePath)
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "TTS Engine အဆင်သင့်ဖြစ်ပါပြီ", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Engine ဖွင့်ရတာ မအောင်မြင်ပါ: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        speakButton.setOnClickListener {
            val text = inputText.text.toString().trim()
            if (text.isNotEmpty()) {
                val session = ortSession
                val env = ortEnv
                if (session == null || env == null) {
                    Toast.makeText(this, "Engine မနိုးသေးပါ၊ ခဏစောင့်ပါ...", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                Toast.makeText(this, "အသံဖိုင်ပြောင်းလဲနေပါသည်...", Toast.LENGTH_SHORT).show()

                thread(start = true) {
                    try {
                        // vocabMap ထဲက ID အစစ်အမှန်များကို ကွက်တိပြောင်းလဲခြင်း (မရှိပါက ၅၇ ဟုသတ်မှတ်)
                        val inputSequence = LongArray(text.length) { i ->
                            vocabMap[text[i]] ?: 57L
                        }
                        val inputShape = longArrayOf(1, inputSequence.size.toLong())

                        // Input Tensors ပြင်ဆင်ခြင်း
                        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                        
                        val maskSequence = LongArray(text.length) { 1L }
                        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(maskSequence), inputShape)
                        
                        val speakerSequence = longArrayOf(0L)
                        val speakerShape = longArrayOf(1)
                        val speakerTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(speakerSequence), speakerShape)

                        // မော်ဒယ်မှ တောင်းဆိုသော အမည်များအတိုင်း ကွက်တိထည့်သွင်းခြင်း
                        val inputMap = HashMap<String, OnnxTensor>()
                        inputMap["input"] = inputTensor
                        inputMap["input_lengths"] = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), longArrayOf(1))
                        inputMap["attention_mask"] = maskTensor
                        inputMap["speaker_id"] = speakerTensor
                        inputMap["sid"] = speakerTensor

                        // VITS Model တောင်းဆိုတတ်သော default input name အားလုံးကို map အုပ်ပေးခြင်း
                        val inputNames = session.inputNames
                        for (name in inputNames) {
                            if (!inputMap.containsKey(name)) {
                                when {
                                    name.contains("input_lengths") || name.contains("scales") -> {
                                        // scale configuration အတွက် float array သို့မဟုတ် long array အစားထိုးခြင်း
                                    }
                                    name.contains("input") -> inputMap[name] = inputTensor
                                    name.contains("mask") -> inputMap[name] = maskTensor
                                    name.contains("speaker") || name.contains("sid") -> inputMap[name] = speakerTensor
                                }
                            }
                        }

                        // စက်ရုပ်ထဲ ကုဒ်ပတ်ခြင်း
                        val results = session.run(inputMap)
                        val outputTensor = results.get(0) as OnnxTensor
                        
                        val floatBuffer = outputTensor.floatBuffer
                        val audioFloats = FloatArray(floatBuffer.remaining())
                        floatBuffer.get(audioFloats)
                        
                        // မော်ဒယ်ရဲ့ config.json အရ sampling_rate ကို 16000 သို့ ပြောင်းလဲခြင်း
                        val sampleRate = 16000 
                        val bufferSize = AudioTrack.getMinBufferSize(
                            sampleRate,
                            AudioFormat.CHANNEL_OUT_MONO,
                            AudioFormat.ENCODING_PCM_FLOAT
                        )
                        
                        val audioTrack = AudioTrack(
                            AudioManager.STREAM_MUSIC,
                            sampleRate,
                            AudioFormat.CHANNEL_OUT_MONO,
                            AudioFormat.ENCODING_PCM_FLOAT,
                            bufferSize,
                            AudioTrack.MODE_STREAM
                        )
                        
                        audioTrack.play()
                        audioTrack.write(audioFloats, 0, audioFloats.size, AudioTrack.WRITE_BLOCKING)
                        
                        // Tensor များ ပိတ်သိမ်းခြင်း
                        inputTensor.close()
                        maskTensor.close()
                        speakerTensor.close()
                        results.close()

                    } catch (e: Exception) {
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, "အသံထုတ်လုပ်မှု အမှားတက်သွားပါသည်: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } else {
                Toast.makeText(this, "စာသား အရင်ရိုက်ပေးပါ", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortSession?.close()
        ortEnv?.close()
    }
}
