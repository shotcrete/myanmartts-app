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

        // ၁။ ONNX Runtime Engine ကို နောက်ကွယ်မှာ နှိုးခြင်း
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("model.onnx").readBytes()
            ortSession = ortEnv?.createSession(modelBytes)
            Toast.makeText(this, "TTS Engine အဆင်သင့်ဖြစ်ပါပြီ", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Engine ဖွင့်ရတာ မအောင်မြင်ပါ: ${e.message}", Toast.LENGTH_LONG).show()
        }

        speakButton.setOnClickListener {
            val text = inputText.text.toString().trim()
            if (text.isNotEmpty()) {
                Toast.makeText(this, "အသံဖိုင်ပြောင်းလဲနေပါသည်...", Toast.LENGTH_SHORT).show()

                // ၂။ ဖုန်းမလေးသွားအောင် အသံထုတ်လုပ်ရေးလုပ်ငန်းကို နောက်ကွယ် Thread နဲ့ ခိုင်းခြင်း
                thread(start = true) {
                    try {
                        val session = ortSession
                        val env = ortEnv
                        if (session != null && env != null) {
                            
                            // စာသားကို ယူပြီး Input Tensor အဖြစ် ပြောင်းလဲခြင်း
                            // (မှတ်ချက် - မော်ဒယ်ဖွဲ့စည်းပုံအပေါ်မူတည်၍ LongArray သို့မဟုတ် IntArray လိုအပ်နိုင်သည်)
                            val inputSequence = LongArray(text.length) { i -> text[i].code.toLong() }
                            val inputShape = longArrayOf(1, inputSequence.size.toLong())
                            
                            val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                            
                            // မော်ဒယ်ရဲ့ Input Name ကို ယူခြင်း (ပုံမှန်အားဖြင့် "input" သို့မဟုတ် ပထမဆုံး key)
                            val inputName = session.inputNames.iterator().next()
                            
                            // ၃။ ပင်မ Engine ထဲ ကုဒ်ပတ်ပြီး အသံလှိုင်းထုတ်ယူခြင်း (Inference)
                            val results = session.run(Collections.singletonMap(inputName, inputTensor))
                            val outputTensor = results.get(0) as OnnxTensor
                            
                            // Float Array (PCM Audio အစိမ်း) အဖြစ် ထုတ်ယူခြင်း
                            val audioFloats = outputTensor.floatBuffer.array()
                            
                            // ၄။ စပီကာဆီ လမ်းကြောင်းဖွင့်ပြီး အသံ အတင်းအော်ခိုင်းခြင်း (AudioTrack ဖွင့်ခြင်း)
                            // မြန်မာ TTS Model အများစုသည် 22050Hz သို့မဟုတ် 16000Hz သုံးတတ်ကြသည်
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
