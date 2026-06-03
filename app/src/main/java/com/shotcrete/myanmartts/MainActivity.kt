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
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // vocab.json အတိုင်း ကွက်တိသတ်မှတ်ထားသော ID များ
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

        // ၁။ ONNX Core Engine အား နောက်ကွယ်မှ နှိုးခြင်း
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
                        // Python သတ်မှတ်ချက်အတိုင်း Space ဖယ်ထုတ်ခြင်း
                        val cleanText = text.replace(" ", "")
                        
                        // ၂။ Hugging Face တရားဝင် VitsTokenizer အလုပ်လုပ်ပုံအတိုင်း Interspersed Pad (0L) စနစ်တည်ဆောက်ခြင်း
                        // စာလုံးတစ်လုံးစီ၏ ကြားထဲတွင် Pad Token (0L) ကို ညှပ်ထည့်ပေးရပါသည်
                        val tokenList = mutableListOf<Long>()
                        tokenList.add(0L) // ရှေ့ဆုံး Pad
                        for (i in cleanText.indices) {
                            val id = vocabMap[cleanText[i]] ?: 56L // မတွေ့ပါက space id ပေးမည်
                            tokenList.add(id)
                            tokenList.add(0L) // ကြား/နောက် Pad
                        }

                        val inputSequence = tokenList.toLongArray()
                        val inputShape = longArrayOf(1, inputSequence.size.toLong())

                        // Tensors များ အသင့်ပြင်ဆင်ခြင်း
                        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                        val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), longArrayOf(1))
                        
                        val maskSequence = LongArray(inputSequence.size) { 1L }
                        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(maskSequence), inputShape)
                        val speakerTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(0L)), longArrayOf(1))

                        // မော်ဒယ်လိုအပ်ချက်များကို Dynamic စစ်ထုတ်ထည့်သွင်းခြင်း
                        val inputMap = HashMap<String, OnnxTensor>()
                        val expectedInputNames = session.inputNames

                        for (name in expectedInputNames) {
                            when {
                                name.contains("input_lengths") -> inputMap[name] = lengthTensor
                                name == "input" || name.contains("input_ids") -> inputMap[name] = inputTensor
                                name.contains("attention_mask") || name.contains("mask") -> inputMap[name] = maskTensor
                                name.contains("speaker_id") || name.contains("sid") -> inputMap[name] = speakerTensor
                            }
                        }

                        // ၃။ Model Inference မောင်းနှင်ခြင်း
                        val results = session.run(inputMap)
                        val outputTensor = results.get(0) as OnnxTensor
                        
                        val floatBuffer = outputTensor.floatBuffer
                        val audioFloats = FloatArray(floatBuffer.remaining())
                        floatBuffer.get(audioFloats)

                        // ၄။ အသံတိုးခြင်းနှင့် အရှိန်လွန်မြန်ခြင်းတို့အား ဒေတာပြုပြင်ထိန်းချုပ်မှုဖြင့် ကုစားခြင်း
                        // Float Waveform အား စံချိန်ကိုက် အသံမြှင့်တင်ပေးခြင်း (Volume Normalization)
                        var maxVal = 0.0f
                        for (f in audioFloats) {
                            val absF = if (f < 0) -f else f
                            if (absF > maxVal) maxVal = absF
                        }
                        
                        // အသံတိုးလွန်းခြင်းမှ ကာကွယ်ရန် သင့်တော်သော Gain နှုန်းဖြင့် မြှင့်တင်ခြင်း
                        if (maxVal > 0) {
                            val gain = 0.8f / maxVal
                            for (i in audioFloats.indices) {
                                audioFloats[i] = audioFloats[i] * gain
                            }
                        }

                        // ၅။ ၁၆၀၀၀ Hz စနစ်ဖြင့် အသံလွှင့်ထုတ်ခြင်း
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
                        
                        // ပိတ်သိမ်းခြင်း
                        inputTensor.close()
                        lengthTensor.close()
                        maskTensor.close()
                        speakerTensor.close()
                        results.close()

                    } catch (e: Exception) {
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, "အမှားတက်သွားပါသည်: ${e.message}", Toast.LENGTH_LONG).show()
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
