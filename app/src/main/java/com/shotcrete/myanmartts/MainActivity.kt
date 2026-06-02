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
import java.util.Collections
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val inputText = findViewById<EditText>(R.id.inputText)
        val speakButton = findViewById<Button>(R.id.speakButton)

        // ၁။ ဖုန်း RAM မပြည့်အောင် Model ဖိုင်ကြီးကို နောက်ကွယ် Thread ဖြင့် Storage ထဲအရင်ရွှေ့ပြီးမှ Engine နှိုးခြင်း
        thread(start = true) {
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                
                // Storage ထဲမှာ model.onnx ရှိမရှိစစ်၊ မရှိရင် Assets မှ ကူးထည့်ခြင်း
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

                // လမ်းကြောင်းအမှန်မှ Session ကို တည်ဆောက်ခြင်း
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
                if (session == null) {
                    Toast.makeText(this, "Engine မက်မွန်သီး မနိုးသေးပါ၊ ခဏစောင့်ပါ...", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                Toast.makeText(this, "အသံဖိုင်ပြောင်းလဲနေပါသည်...", Toast.LENGTH_SHORT).show()

                // ၂။ အသံထုတ်လုပ်ရေးလုပ်ငန်းကို နောက်ကွယ် Thread နဲ့ ခိုင်းခြင်း
                thread(start = true) {
                    try {
                        val env = ortEnv
                        if (env != null) {
                            
                            // စာသားကို မော်ဒယ်နားလည်သော ID Range ထဲရောက်အောင် ဘာသာပြန်ပေးခြင်း
                            val inputSequence = LongArray(text.length) { i -> 
                                val code = text[i].code
                                if (code in 4096..4255) {
                                    (code - 4120).toLong() 
                                } else {
                                    (code % 50).toLong() 
                                }
                            }
                            val inputShape = longArrayOf(1, inputSequence.size.toLong())
                            
                            val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                            val inputName = session.inputNames.iterator().next()
                            
                            // ၃။ ပင်မ Engine ထဲ ကုဒ်ပတ်ပြီး အသံလှိုင်းထုတ်ယူခြင်း (Inference)
                            val results = session.run(Collections.singletonMap(inputName, inputTensor))
                            val outputTensor = results.get(0) as OnnxTensor
                            
                            // Float Array (PCM Audio) ကို ဖတ်ယူခြင်း
                            val floatBuffer = outputTensor.floatBuffer
                            val audioFloats = FloatArray(floatBuffer.remaining())
                            floatBuffer.get(audioFloats)
                            
                            // ၄။ စပီကာဆီ လမ်းကြောင်းဖွင့်ပြီး အသံ ကစားခြင်း (AudioTrack)
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
                            
                            // အသံစတင်ကစားပြီ!
                            audioTrack.play()
                            audioTrack.write(audioFloats, 0, audioFloats.size, AudioTrack.WRITE_BLOCKING)
                            
                            // အလုပ်ပြီးရင် ပိတ်သိမ်းခြင်း
                            inputTensor.close()
                            results.close()
                        }
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

