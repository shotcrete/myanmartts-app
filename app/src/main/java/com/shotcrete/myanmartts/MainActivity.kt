package com.shotcrete.myanmartts

import android.app.ProgressDialog
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Environment
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.HashMap
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var progressDialog: ProgressDialog? = null
    
    // နောက်ဆုံးထွက်ထားသော အသံဖိုင်လမ်းကြောင်းကို မှတ်ထားရန်
    private var lastAudioFilePath: String? = null
    private var mediaPlayer: MediaPlayer? = null

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
        val playLastButton = findViewById<Button>(R.id.playLastButton)

        progressDialog = ProgressDialog(this).apply {
            setMessage("မြန်မာအသံဖိုင်ပြောင်းလဲနေပါသည်... ခဏစောင့်ပါ...")
            setCancelable(false)
        }

        // Engine နှိုးခြင်း
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
                    Toast.makeText(this@MainActivity, "Engine Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        speakButton.setOnClickListener {
            val rawText = inputText.text.toString().trim()
            if (rawText.isNotEmpty()) {
                val session = ortSession
                val env = ortEnv
                if (session == null || env == null) {
                    Toast.makeText(this, "Engine မနိုးသေးပါ...", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                progressDialog?.show()

                thread(start = true) {
                    try {
                        // 🛠️ ၁။ စာလုံးကျော်ခြင်းနှင့် အသံပျက်ခြင်းကို ကာကွယ်ရန် စာသားကြိုတင်ပြင်ဆင်ခြင်း
                        var cleanedText = preProcessMyanmarText(rawText)
                        cleanedText = normalizeNumbers(cleanedText)
                        cleanedText = cleanedText.replace(" ", "")

                        val chunks = splitTextIntoChunks(cleanedText)
                        val combinedAudioList = mutableListOf<FloatArray>()

                        for (chunk in chunks) {
                            val validChars = chunk.filter { vocabMap.containsKey(it) }
                            if (validChars.length < 2) continue

                            val tokenList = mutableListOf<Long>()
                            tokenList.add(0L)
                            for (i in validChars.indices) {
                                val id = vocabMap[validChars[i]] ?: 56L
                                tokenList.add(id)
                                tokenList.add(0L)
                            }

                            val inputSequence = tokenList.toLongArray()
                            val inputShape = longArrayOf(1, inputSequence.size.toLong())

                            val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                            val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), longArrayOf(1))
                            val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(LongArray(inputSequence.size) { 1L }), inputShape)
                            val speakerTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(0L)), longArrayOf(1))

                            val inputMap = HashMap<String, OnnxTensor>()
                            for (name in session.inputNames) {
                                when {
                                    name.contains("input_lengths") -> inputMap[name] = lengthTensor
                                    name == "input" || name.contains("input_ids") -> inputMap[name] = inputTensor
                                    name.contains("attention_mask") || name.contains("mask") -> inputMap[name] = maskTensor
                                    name.contains("speaker_id") || name.contains("sid") -> inputMap[name] = speakerTensor
                                }
                            }

                            val results = session.run(inputMap)
                            val outputTensor = results.get(0) as OnnxTensor
                            val floatBuffer = outputTensor.floatBuffer
                            val audioFloats = FloatArray(floatBuffer.remaining())
                            floatBuffer.get(audioFloats)

                            combinedAudioList.add(audioFloats)

                            inputTensor.close()
                            lengthTensor.close()
                            maskTensor.close()
                            speakerTensor.close()
                            results.close()
                        }

                        if (combinedAudioList.isEmpty()) {
                            runOnUiThread { progressDialog?.dismiss() }
                            return@thread
                        }
                        
                        val totalLength = combinedAudioList.sumOf { it.size }
                        val finalAudioFloats = FloatArray(totalLength)
                        var destPos = 0
                        for (audioChunk in combinedAudioList) {
                            System.arraycopy(audioChunk, 0, finalAudioFloats, destPos, audioChunk.size)
                            destPos += audioChunk.size
                        }

                        // Volume Normalization
                        var maxVal = 0.0f
                        for (f in finalAudioFloats) {
                            val absF = if (f < 0) -f else f
                            if (absF > maxVal) maxVal = absF
                        }
                        if (maxVal > 0) {
                            val gain = 0.85f / maxVal
                            for (i in finalAudioFloats.indices) {
                                finalAudioFloats[i] = finalAudioFloats[i] * gain
                            }
                        }

                        // 🛠️ ၂။ /sdcard/MyanmarTTS/ ဖိုဒါပုံစံအတိုင်း သီးသန့်နေရာတွင် ဖိုင်ထုတ်ခြင်း
                        val sampleRate = 16000
                        val baseDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC)
                        val myanmarTtsDir = File(baseDir, "MyanmarTTS")
                        if (!myanmarTtsDir.exists()) {
                            myanmarTtsDir.mkdirs()
                        }
                        
                        val outputFile = File(myanmarTtsDir, "TTS_${System.currentTimeMillis()}.wav")
                        saveFloatsToWav(outputFile, finalAudioFloats, sampleRate)

                        // နောက်ဆုံးထွက်ဖိုင်လမ်းကြောင်းကို သိမ်းဆည်းခြင်း
                        lastAudioFilePath = outputFile.absolutePath

                        runOnUiThread {
                            progressDialog?.dismiss()
                            Toast.makeText(this@MainActivity, "ဖိုင်ကို Music/MyanmarTTS/ တွင် သိမ်းပြီးပါပြီ", Toast.LENGTH_LONG).show()
                        }

                        // အသံတိုက်ရိုက်လွှင့်ထုတ်ခြင်း
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
                        audioTrack.write(finalAudioFloats, 0, finalAudioFloats.size, AudioTrack.WRITE_BLOCKING)

                    } catch (e: Exception) {
                        runOnUiThread {
                            progressDialog?.dismiss()
                            Toast.makeText(this@MainActivity, "အမှားတက်သွားပါသည်: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } else {
                Toast.makeText(this, "စာသား အရင်ရိုက်ပေးပါ", Toast.LENGTH_SHORT).show()
            }
        }

        // 🛠️ ၃။ နောက်ဆုံးထွက်ထားသော အသံဖိုင်ကို ပြန်လည်နားထောင်သည့် လုပ်ဆောင်ချက်
        playLastButton.setOnClickListener {
            val path = lastAudioFilePath
            if (path != null && File(path).exists()) {
                try {
                    mediaPlayer?.release()
                    mediaPlayer = MediaPlayer().apply {
                        setDataSource(path)
                        prepare()
                        start()
                    }
                    Toast.makeText(this, "နောက်ဆုံးထွက်ဖိုင်ကို ပြန်ဖွင့်နေပါသည်...", Toast.LENGTH_SHORT).show()
                } catch (e: Exception) {
                    Toast.makeText(this, "ဖွင့်ရန် အဆင်မပြေပါ: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, "ဖွင့်ရန် အသံဖိုင် မရှိသေးပါ၊ အရင်ပြောင်းပေးပါ", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // 🛠️ ၄။ စာလုံးကျော်ဖတ်ခြင်းနှင့် "အီ" သံထွက်ခြင်းကို ဖြေရှင်းပေးမည့် စာသားပြုပြင်စနစ်
    private fun preProcessMyanmarText(text: String): String {
        var res = text
        // မော်ဒယ်ဖတ်ရခက်ခဲသော သင်္ကေတများနှင့် xxxx များကို ဖယ်ရှားခြင်း
        res = res.replace(Regex("[xX._\\-*#+=()_]"), "")
        
        // ၌၊ ၍၊ ၏၊ ဤ စသည်တို့ကို အသံထွက်အတိုင်း စာသားပြောင်းပေးခြင်း
        res = res.replace("၌", "နှိုက်")
        res = res.replace("၍", "ရွေ့")
        res = res.replace("၏", "အိ")
        res = res.replace("ဤ", "အီ")
        
        // ဗျည်းဆင့်စာလုံးများကို မော်ဒယ်ဖတ်နိုင်အောင် အသံထွက်အတိုင်း ယာယီဖြန့်ပေးခြင်း
        res = res.replace("ဘဏ္ဍာ", "ဘန်ဒါ")
        res = res.replace("သဏ္ဌာန်", "သန်ထန်")
        res = res.replace("ဥက္ကာမြဲ", "အုတ်ကာမြဲ")
        
        return res
    }

    private fun splitTextIntoChunks(text: String): List<String> {
        val chunks = mutableListOf<String>()
        val regex = Regex("([^။၊\\n]+[။၊\\n]?)")
        val matches = regex.findAll(text)
        
        var currentChunk = ""
        for (match in matches) {
            val segment = match.value
            if (currentChunk.length + segment.length > 35) {
                if (currentChunk.isNotEmpty()) chunks.add(currentChunk)
                currentChunk = segment
            } else {
                currentChunk += segment
            }
        }
        if (currentChunk.isNotEmpty()) chunks.add(currentChunk)
        return if (chunks.isEmpty()) listOf(text) else chunks
    }

    private fun normalizeNumbers(text: String): String {
        var res = text
        val numMap = mapOf(
            "0" to "သုည", "1" to "တစ်", "2" to "နှစ်", "3" to "သုံး", "4" to "လေး", "5" to "ငါး",
            "6" to "ခြောက်", "7" to "ခုနစ်", "8" to "ရှစ်", "9" to "ကိုး",
            "၀" to "သုည", "၁" to "တစ်", "၂" to "နှစ်", "၃" to "သုံး", "၄" to "လေး", "၅" to "ငါး",
            "၆" to "ခြောက်", "၇" to "ခုနစ်", "၈" to "ရှစ်", "၉" to "ကိုး"
        )
        for ((num, txt) in numMap) {
            res = res.replace(num, txt)
        }
        return res
    }

    private fun saveFloatsToWav(file: File, floatData: FloatArray, sampleRate: Int) {
        val payloadSize = floatData.size * 4 
        val totalSize = payloadSize + 36

        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0) 

            raf.writeBytes("RIFF")
            raf.writeInt(Integer.reverseBytes(totalSize))
            raf.writeBytes("WAVE")

            raf.writeBytes("fmt ")
            raf.writeInt(Integer.reverseBytes(16)) 
            
            raf.writeShort(Integer.reverseBytes(3) shr 16 or (Integer.reverseBytes(3) and 0xFFFF)) 
            raf.writeShort(Integer.reverseBytes(1) shr 16 or (Integer.reverseBytes(1) and 0xFFFF)) 
            
            raf.writeInt(Integer.reverseBytes(sampleRate)) 
            raf.writeInt(Integer.reverseBytes(sampleRate * 4)) 
            
            raf.writeShort(Integer.reverseBytes(4) shr 16 or (Integer.reverseBytes(4) and 0xFFFF)) 
            raf.writeShort(Integer.reverseBytes(32) shr 16 or (Integer.reverseBytes(32) and 0xFFFF)) 

            raf.writeBytes("data")
            raf.writeInt(Integer.reverseBytes(payloadSize))

            val byteBuffer = ByteBuffer.allocate(payloadSize).order(ByteOrder.LITTLE_ENDIAN)
            for (f in floatData) {
                byteBuffer.putFloat(f)
            }
            raf.write(byteBuffer.array())
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mediaPlayer?.release()
        progressDialog?.dismiss()
        ortSession?.close()
        ortEnv?.close()
    }
}
