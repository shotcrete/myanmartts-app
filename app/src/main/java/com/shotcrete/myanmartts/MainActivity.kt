package com.shotcrete.myanmartts

import android.app.ProgressDialog
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
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
import java.io.FileInputStream
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

        // TTS Engine မောင်းနှင်ခြင်း
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
                        // စာသားကြိုတင်သန့်စင်ခြင်း
                        var processedText = preProcessMyanmarText(rawText)
                        processedText = normalizeNumbers(processedText)

                        // 🛠️ ပြုပြင်ချက်- Space နှင့် ပုဒ်ဖြတ်မှုများကို အခြေခံ၍ စနစ်တကျ ဖြတ်တောက်ခြင်း
                        val chunks = splitTextByPunctuationAndSpace(processedText)
                        val combinedAudioList = mutableListOf<FloatArray>()

                        for (chunk in chunks) {
                            val cleanChunk = chunk.replace(" ", "") // မော်ဒယ်ထဲမထည့်ခင် space ဖြုတ်သည်
                            val validChars = cleanChunk.filter { vocabMap.containsKey(it) }
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

                        val sampleRate = 16000
                        val baseDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC)
                        val myanmarTtsDir = File(baseDir, "MyanmarTTS")
                        if (!myanmarTtsDir.exists()) {
                            myanmarTtsDir.mkdirs()
                        }
                        
                        // ယာယီ WAV ဖိုင်ဆောက်ခြင်း
                        val tempWavFile = File(cacheDir, "temp.wav")
                        saveFloatsToWav(tempWavFile, finalAudioFloats, sampleRate)

                        // 🛠️ ပြုပြင်ချက်- WAV မှ သေးငယ်ကျစ်လျစ်သော AAC (.m4a) ဖော်မတ်သို့ တိုက်ရိုက်ပြောင်းလဲသိမ်းဆည်းခြင်း
                        val outputAacFile = File(myanmarTtsDir, "TTS_${System.currentTimeMillis()}.m4a")
                        convertWavToAac(tempWavFile, outputAacFile)
                        tempWavFile.delete() // ယာယီဖိုင်အားဖျက်ပယ်ခြင်း

                        lastAudioFilePath = outputAacFile.absolutePath

                        runOnUiThread {
                            progressDialog?.dismiss()
                            Toast.makeText(this@MainActivity, "AAC (.m4a) ဖိုင်ကို Music/MyanmarTTS/ တွင် သိမ်းပြီးပါပြီ", Toast.LENGTH_LONG).show()
                        }

                        // ဖုန်းထဲတွင် ချက်ချင်းဖွင့်ပြခြင်း
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

        // နောက်ဆုံးထွက်ဖိုင် ပြန်နားထောင်ခြင်း
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

    // 🛠️ ပြုပြင်ချက်- "၏" သံကို "၍" သံမထွက်စေဘဲ မှန်ကန်အောင် အသံဖလှယ်ပေးသည့်စနစ်
    private fun preProcessMyanmarText(text: String): String {
        var res = text
        res = res.replace(Regex("[xX._\\-*#+=()_]"), "")
        
        // "၏" အသံကို "အစ်" သို့မဟုတ် စကားစပ်အလိုက် "အဲ" သံထွက်စေရန် ပြောင်းလဲခြင်း
        res = res.replace("၏", "အစ်")
        res = res.replace("၌", "နှိုက်")
        res = res.replace("၍", "ရွေ့")
        res = res.replace("ဤ", "အီ")
        
        res = res.replace("ဘဏ္ဍာ", "ဘန်ဒါ")
        res = res.replace("သဏ္ဌာန်", "သန်ထန်")
        res = res.replace("ဥက္ကာမြဲ", "အုတ်ကာမြဲ")
        
        return res
    }

    // 🛠️ ပြုပြင်ချက်- စကားစုအလယ်ခေါင်တွင် အလိုအလျောက်ပြတ်မသွားစေဘဲ Space နှင့် ပုဒ်ဖြတ်များအတိုင်း တိကျစွာခွဲထုတ်သည့်စနစ်
    private fun splitTextByPunctuationAndSpace(text: String): List<String> {
        val chunks = mutableListOf<String>()
        // ပုဒ်မ(။)၊ ကော်မာ(၊)၊ Space( ) နှင့် စာကြောင်းအသစ်(\n) များကိုသာ အခြေခံ၍ တိကျစွာ ဖြတ်တောက်သည်
        val regex = Regex("([^။၊ \\n]+[။၊ \\n]?)")
        val matches = regex.findAll(text)
        
        var currentChunk = ""
        for (match in matches) {
            val segment = match.value
            // စကားစုအလိုက် စုစည်းပြီးမှ စာလုံးရေ ၅၀ ကျော်မှသာ နောက်တစ်ပိုင်း ခွဲထုတ်သည်
            if (currentChunk.length + segment.length > 50) {
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

    // 🛠️ စွမ်းဆောင်ရည်မြင့်မားပြီး အသံဖိုင်အရွယ်အစားကို အဆမတန် သေးငယ်စေမည့် WAV to AAC (.m4a) Converter စနစ်
    private fun convertWavToAac(wavFile: File, aacFile: File) {
        val fis = FileInputStream(wavFile)
        // WAV Header 44 bytes ကို ကျော်ခွပစ်ခြင်း
        fis.skip(44)

        val fos = FileOutputStream(aacFile)
        
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, 16000, 1).apply {
            setInteger(MediaFormat.KEY_BIT_RATE, 64000)
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 1024 * 10)
        }
        
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()

        val inputBuffers = codec.inputBuffers
        val outputBuffers = codec.outputBuffers
        val bufferInfo = MediaCodec.BufferInfo()
        
        val rawBuffer = ByteArray(4 * 1024)
        var hasMoreData = true
        var isEOS = false

        while (!isEOS) {
            if (hasMoreData) {
                val inputBufferIndex = codec.dequeueInputBuffer(10000)
                if (inputBufferIndex >= 0) {
                    val inputBuffer = inputBuffers[inputBufferIndex]
                    inputBuffer.clear()
                    val bytesRead = fis.read(rawBuffer)
                    if (bytesRead == -1) {
                        hasMoreData = false
                        codec.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    } else {
                        inputBuffer.put(rawBuffer, 0, bytesRead)
                        codec.queueInputBuffer(inputBufferIndex, 0, bytesRead, 0, 0)
                    }
                }
            }

            var outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 10000)
            while (outputBufferIndex >= 0) {
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    isEOS = true
                }
                val outputBuffer = outputBuffers[outputBufferIndex]
                outputBuffer.position(bufferInfo.offset)
                outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                
                // ADTS Header လောင်းထည့်ပြီး AAC သို့ တိုက်ရိုက်ပြောင်းလဲသိမ်းဆည်းခြင်း
                val outBitsSize = bufferInfo.size
                val packetSize = outBitsSize + 7
                val adtsHeader = ByteArray(7)
                adtsHeader[0] = 0xFF.toByte()
                adtsHeader[1] = 0xF1.toByte()
                adtsHeader[2] = (((0 certification 1) shl 6) + ( certification 4 shl 2) + (1 shl 1)).toByte()
                adtsHeader[3] = (((1 shl 6) + (packetSize shr 11)).toByte())
                adtsHeader[4] = ((packetSize and 0x7FF) shr 3).toByte()
                adtsHeader[5] = (((packetSize and 7) shl 5) + 0x1F).toByte()
                adtsHeader[6] = 0xFC.toByte()

                fos.write(adtsHeader)
                val outData = ByteArray(bufferInfo.size)
                outputBuffer.get(outData)
                fos.write(outData)

                codec.releaseOutputBuffer(outputBufferIndex, false)
                outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 0)
            }
        }

        codec.stop()
        codec.release()
        fis.close()
        fos.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        mediaPlayer?.release()
        progressDialog?.dismiss()
        ortSession?.close()
        ortEnv?.close()
    }
}
