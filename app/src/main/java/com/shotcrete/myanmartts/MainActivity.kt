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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val inputText = findViewById<EditText>(R.id.inputText)
        val speakButton = findViewById<Button>(R.id.speakButton)

        // ၁။ နောက်ကွယ် Thread ဖြင့် Storage ထဲ Cache ရွှေ့ပြီး Engine နှိုးခြင်း
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

                // ၂။ အသံထုတ်လုပ်ရေးလုပ်ငန်းကို သီးသန့် Thread ဖြင့် လုပ်ဆောင်ခြင်း
                thread(start = true) {
                    try {
                        // စာသားအား မော်ဒယ်နားလည်သော ID Range သို့ ပြောင်းလဲခြင်း
                        val inputSequence = LongArray(text.length) { i -> 
                            val code = text[i].code
                            if (code in 4096..4255) {
                                (code - 4120).toLong() 
                            } else {
                                (code % 50).toLong() 
                            }
                        }
                        val inputShape = longArrayOf(1, inputSequence.size.toLong())

                        // မော်ဒယ်လိုအပ်သော Input (၃) မျိုးစလုံးကို ပြင်ဆင်ခြင်း
                        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                        
                        val maskSequence = LongArray(text.length) { 1L }
                        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(maskSequence), inputShape)
                        
                        val speakerSequence = longArrayOf(0L)
                        val speakerShape = longArrayOf(1)
                        val speakerTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(speakerSequence), speakerShape)

                        // ဒေတာအားလုံးကို Map ဖြင့် စုံလင်စွာ ပေါင်းစပ်ခြင်း
                        val inputMap = HashMap<String, OnnxTensor>()
                        val inputNames = session.inputNames
                        for (name in inputNames) {
                            when {
                                name.contains("input") -> inputMap[name] = inputTensor
                                name.contains("mask") -> inputMap[name] = maskTensor
                                name.contains("speaker") || name.contains("sid") -> inputMap[name] = speakerTensor
                            }
                        }
                        
                        if (!inputMap.containsKey("attention_mask") && inputNames.contains("attention_mask")) {
                            inputMap["attention_mask"] = maskTensor
                        }

                        // ၃။ ပင်မ Engine ထဲ ကုဒ်ပတ်ခြင်း (Inference)
                        val results = session.run(inputMap)
                        val outputTensor = results.get(0) as OnnxTensor
                        
                        // Float Array (PCM Audio) ကို ဖတ်ယူခြင်း
                        val floatBuffer = outputTensor.floatBuffer
                        val audioFloats = FloatArray(floatBuffer.remaining())
                        floatBuffer.get(audioFloats)
                        
                        // ၄။ AudioTrack စနစ်ဖြင့် ဖုန်းစပီကာမှ အသံလွှင့်ထုတ်ခြင်း
                        val sampleRate = 22050 
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
                        
                        // သုံးပြီးသား Tensor များ ပြန်ပိတ်ခြင်း
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
