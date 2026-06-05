package com.shotcrete.myanmartts

import android.app.ProgressDialog
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.HashMap
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment? = null
    private var ortSessionMm: OrtSession? = null
    private var ortSessionEn: OrtSession? = null
    private var progressDialog: ProgressDialog? = null
    
    private var lastAudioFilePath: String? = null
    private var mediaPlayer: MediaPlayer? = null

    // မြန်မာ Vocab Map
    private val vocabMapMm = mapOf(
        '်' to 0L, 'ာ' to 1L, 'ု' to 2L, 'ိ' to 3L, 'း' to 4L, 'ေ' to 5L, 'သ' to 6L, 'က' to 7L,
        'င' to 8L, 'တ' to 9L, '့' to 10L, 'မ' to 11L, 'ြ' to 12L, 'ည' to 13L, 'ရ' to 14L, 'အ' to 15L,
        'န' to 16L, 'လ' to 17L, 'ှ' to 18L, 'ပ' to 19L, 'စ' to 20L, 'ခ' to 21L, 'ျ' to 22L, 'ူ' to 23L,
        'ွ' to 24L, 'ါ' to 25L, 'ထ' to 26L, 'ဖ' to 27L, 'ံ' to 28L, 'ယ' to 29L, 'ဆ' to 30L, 'ီ' to 31L,
        'ဲ' to 32L, 'ဟ' to 33L, 'ဘ' to 34L, 'ဝ' to 35L, '္' to 36L, 'ဉ' to 37L, 'ဤ' to 38L, 'ဇ' to 39L,
        'ဒ' to 40L, 'ဂ' to 41L, 'ဦ' to 42L, 'ဏ' to 43L, 'ဗ' to 44L, 'ဓ' to 45L, 'ဧ' to 46L, 'ဥ' to 47L,
        'ဩ' to 48L, 'ဌ' to 49L, 'ဋ' to 50L, '\'' to 51L, 'ဣ' to 52L, 'ဍ' to 53L, 'ဿ' to 54L, 'ဈ' to 55L,
        ' ' to 56L
    )

    // အင်္ဂလိပ် Vocab Map
    private val vocabMapEn = mapOf(
        '_' to 0L, '^' to 1L, '$' to 2L, ' ' to 3L, '!' to 4L, '\'' to 5L, ',' to 6L, '-' to 7L,
        '.' to 8L, ';' to 9L, '?' to 10L, 'a' to 11L, 'b' to 12L, 'd' to 13L, 'e' to 14L, 'f' to 15L,
        'h' to 16L, 'i' to 17L, 'j' to 18L, 'k' to 19L, 'l' to 20L, 'm' to 21L, 'n' to 22L, 'o' to 23L,
        'p' to 24L, 'r' to 25L, 's' to 26L, 't' to 27L, 'u' to 28L, 'v' to 29L, 'w' to 30L, 'z' to 31L,
        'æ' to 32L, 'ç' to 33L, 'ð' to 34L, 'ø' to 35L, 'ŋ' to 36L, 'ɐ' to 37L, 'ɒ' to 38L, 'ɔ' to 39L,
        'ə' to 40L, 'ɛ' to 41L, 'ɜ' to 42L, 'ɪ' to 43L, 'ɫ' to 44L, 'ɱ' to 45L, 'œ' to 46L, 'ɒ' to 47L,
        'ʃ' to 48L, 'θ' to 49L, 'ʊ' to 50L, 'ʌ' to 51L, 'ʒ' to 52L, 'θ' to 53L, 'ː' to 54L, 'ˈ' to 55L,
        'ˌ' to 56L
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val inputText = findViewById<EditText>(R.id.inputText)
        val speakButton = findViewById<Button>(R.id.speakButton)
        val playLastButton = findViewById<Button>(R.id.playLastButton)

        progressDialog = ProgressDialog(this).apply {
            setMessage("အသံဖိုင်ပြောင်းလဲနေပါသည်... ขဏစောင့်ပါ...")
            setCancelable(false)
        }

        thread(start = true) {
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                
                val modelFileMm = File(cacheDir, "model.onnx")
                if (!modelFileMm.exists() || modelFileMm.length() < 10_000_000) {
                    if (modelFileMm.exists()) modelFileMm.delete()
                    copyAssetToFile("model.onnx", modelFileMm)
                }
                ortSessionMm = ortEnv?.createSession(modelFileMm.absolutePath)

                val modelFileEn = File(cacheDir, "en_model.onnx")
                if (!modelFileEn.exists() || modelFileEn.length() < 10_000_000) {
                    if (modelFileEn.exists()) modelFileEn.delete()
                    copyAssetToFile("en_model.onnx", modelFileEn)
                }
                
                if (modelFileEn.exists() && modelFileEn.length() > 10_000_000) {
                    ortSessionEn = ortEnv?.createSession(modelFileEn.absolutePath)
                }

                runOnUiThread {
                    Toast.makeText(this@MainActivity, "မြန်မာ + အင်္ဂလိပ် Engine အဆင်သင့်ဖြစ်ပါပြီဗျာ", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Engine Loading Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        speakButton.setOnClickListener {
            val rawText = inputText.text.toString().trim()
            if (rawText.isNotEmpty()) {
                val env = ortEnv
                if (ortSessionMm == null || env == null) {
                    Toast.makeText(this, "မြန်မာ Engine အဆင်သင့်မဖြစ်သေးပါ...", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                progressDialog?.show()

                thread(start = true) {
                    try {
                        val textSegments = splitByLanguage(rawText)
                        val combinedAudioList = mutableListOf<FloatArray>()

                        for (segment in textSegments) {
                            if (segment.isEnglish) {
                                val currentSessionEn = ortSessionEn
                                if (currentSessionEn != null) {
                                    try {
                                        val phonemes = textToPhonemes(segment.text)
                                        val tokenList = mutableListOf<Long>()
                                        
                                        // 🚀 [VITS Core Fix] BOS တိုကင်ကို သတ်မှတ်ချက်အတိုင်း ထည့်သွင်းခြင်း
                                        tokenList.add(1L) 
                                        
                                        // Phonemes တစ်လုံးချင်းစီကို သက်ဆိုင်ရာ Vocab ID သို့ ပြောင်းလဲခြင်း
                                        for (ch in phonemes) { 
                                            // 🚀 [Bug Fix] စာလုံးအလွတ်ဖြစ်နေပါက Space (3L) ဖြင့် အစားထိုးခြင်း
                                            tokenList.add(vocabMapEn[ch] ?: 3L) 
                                        }
                                        
                                        // EOS တိုကင် ထည့်သွင်းခြင်း
                                        tokenList.add(2L) 

                                        val inputSequence = tokenList.toLongArray()
                                        if (inputSequence.size < 3) continue

                                        val inputShape = longArrayOf(1, inputSequence.size.toLong())
                                        val singleShape = longArrayOf(1)

                                        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                                        val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), singleShape)
                                        val scalesTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f)), longArrayOf(3))
                                        
                                        val attentionMaskSequence = LongArray(inputSequence.size) { 1L }
                                        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(attentionMaskSequence), inputShape)
                                        val sidTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(0L)), singleShape)

                                        val inputMap = HashMap<String, OnnxTensor>()
                                        
                                        currentSessionEn.inputNames?.forEach { name ->
                                            val lowerName = name.lowercase()
                                            when {
                                                lowerName.contains("mask") || lowerName.contains("attention") -> inputMap[name] = maskTensor
                                                lowerName.contains("input_ids") || lowerName.contains("input") || lowerName == "text" -> inputMap[name] = inputTensor
                                                lowerName.contains("length") -> inputMap[name] = lengthTensor
                                                lowerName.contains("scale") -> inputMap[name] = scalesTensor
                                                lowerName.contains("sid") || lowerName.contains("speaker") -> inputMap[name] = sidTensor
                                            }
                                        }

                                        val results = currentSessionEn.run(inputMap)
                                        val outputTensor = results?.get(0) as? OnnxTensor
                                        outputTensor?.let {
                                            val floatBuffer = it.floatBuffer
                                            val audioFloats = FloatArray(floatBuffer.remaining())
                                            floatBuffer.get(audioFloats)
                                            combinedAudioList.add(audioFloats)
                                        }
                                        
                                        inputTensor.close()
                                        lengthTensor.close()
                                        scalesTensor.close()
                                        maskTensor.close()
                                        sidTensor.close()
                                        results?.close()
                                    } catch (e: Exception) {
                                        e.printStackTrace()
                                    }
                                } else {
                                    runOnUiThread {
                                        Toast.makeText(this@MainActivity, "အင်္ဂလိပ် Engine အလုပ်မလုပ်သေးပါ", Toast.LENGTH_SHORT).show()
                                    }
                                }
                            } else {
                                runMyanmarPipeline(segment.text, env, combinedAudioList)
                            }
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
                        if (!myanmarTtsDir.exists()) myanmarTtsDir.mkdirs()

                        val tempPcmFile = File(cacheDir, "temp.pcm")
                        saveFloatsToPcm16(tempPcmFile, finalAudioFloats)

                        val outputAacFile = File(myanmarTtsDir, "HYBRID_TTS_${System.currentTimeMillis()}.m4a")
                        convertPcmToAacMuxer(tempPcmFile, outputAacFile, sampleRate)
                        tempPcmFile.delete()

                        lastAudioFilePath = outputAacFile.absolutePath

                        runOnUiThread {
                            progressDialog?.dismiss()
                            Toast.makeText(this@MainActivity, "Hybrid အသံထွက်လာပါပြီဗျာ", Toast.LENGTH_LONG).show()
                        }

                        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
                        val audioTrack = AudioTrack(
                            AudioManager.STREAM_MUSIC, 
                            sampleRate, 
                            AudioFormat.CHANNEL_OUT_MONO, 
                            AudioFormat.ENCODING_PCM_FLOAT, 
                            maxOf(bufferSize, finalAudioFloats.size * 4), 
                            AudioTrack.MODE_STATIC 
                        )
                        audioTrack.write(finalAudioFloats, 0, finalAudioFloats.size, AudioTrack.WRITE_BLOCKING)
                        audioTrack.play()

                    } catch (e: Exception) {
                        e.printStackTrace()
                        runOnUiThread {
                            progressDialog?.dismiss()
                            Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } else {
                Toast.makeText(this, "စာသား အရင်ရိုက်ပေးပါ", Toast.LENGTH_SHORT).show()
            }
        }

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
                } catch (e: Exception) {
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun runMyanmarPipeline(text: String, env: OrtEnvironment, combinedAudioList: MutableList<FloatArray>) {
        var processedText = preProcessMyanmarText(text)
        processedText = normalizeNumbers(processedText)

        val cleanChunk = processedText.replace(" ", "")
        val validChars = cleanChunk.filter { vocabMapMm.containsKey(it) }
        if (validChars.length < 2) return

        val tokenList = mutableListOf<Long>()
        tokenList.add(0L)
        for (i in validChars.indices) {
            val id = vocabMapMm[validChars[i]] ?: 56L
            if (id in 0L..56L) {
                tokenList.add(id)
                tokenList.add(0L)
            }
        }

        val inputSequence = tokenList.toLongArray()
        if (inputSequence.isEmpty()) return

        val inputShape = longArrayOf(1, inputSequence.size.toLong())
        val singleShape = longArrayOf(1)
        
        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
        val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), singleShape)
        val scalesTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f)), longArrayOf(3))
        
        val attentionMaskSequence = LongArray(inputSequence.size) { 1L }
        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(attentionMaskSequence), inputShape)
        val sidTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(0L)), singleShape)

        val inputMap = HashMap<String, OnnxTensor>()
        
        ortSessionMm?.inputNames?.forEach { name ->
            val lowerName = name.lowercase()
            when {
                lowerName.contains("mask") || lowerName.contains("attention") -> inputMap[name] = maskTensor
                name == "input" || lowerName.contains("input_ids") || lowerName == "text" -> inputMap[name] = inputTensor
                lowerName.contains("length") -> inputMap[name] = lengthTensor
                lowerName.contains("scale") -> inputMap[name] = scalesTensor
                lowerName.contains("sid") || lowerName.contains("speaker") -> inputMap[name] = sidTensor
            }
        }

        val results = ortSessionMm?.run(inputMap)
        val outputTensor = results?.get(0) as? OnnxTensor
        outputTensor?.let {
            val floatBuffer = it.floatBuffer
            val audioFloats = FloatArray(floatBuffer.remaining())
            floatBuffer.get(audioFloats)
            combinedAudioList.add(audioFloats)
        }
        inputTensor.close()
        lengthTensor.close()
        scalesTensor.close()
        maskTensor.close()
        sidTensor.close()
        results?.close()
    }

    data class LangSegment(val text: String, val isEnglish: Boolean)
    
    private fun splitByLanguage(text: String): List<LangSegment> {
        val segments = mutableListOf<LangSegment>()
        if (text.isEmpty()) return segments

        var currentSegment = StringBuilder()
        var currentIsEnglish: Boolean? = null

        for (ch in text) {
            val computedIsEnglish = if (ch in " .,!?'-" && currentIsEnglish != null) {
                currentIsEnglish!! 
            } else {
                ch in 'a'..'z' || ch in 'A'..'Z'
            }

            if (currentIsEnglish == null) {
                currentIsEnglish = computedIsEnglish
                currentSegment.append(ch)
            } else if (currentIsEnglish == computedIsEnglish) {
                currentSegment.append(ch)
            } else {
                if (currentSegment.toString().trim().isNotEmpty()) {
                    segments.add(LangSegment(currentSegment.toString(), currentIsEnglish!!))
                }
                currentSegment = StringBuilder().append(ch)
                currentIsEnglish = computedIsEnglish
            }
        }
        if (currentSegment.toString().trim().isNotEmpty()) {
            segments.add(LangSegment(currentSegment.toString(), currentIsEnglish ?: false))
        }
        return segments
    }

    private fun textToPhonemes(text: String): String {
        var raw = text.lowercase(Locale.ROOT)
        
        // 🚀 [Standard Dictionary] စာလုံးအချို့ကို သတ်မှတ်ထားသော Phonemes ပြောင်းလဲခြင်း
        val g2p = mapOf(
            "hello" to "həˈloʊ", "hi" to "ˈhaɪ", "ok" to "oʊˈkeɪ", "good" to "ˈɡʊd",
            "morning" to "ˈmɔːrnɪŋ", "thank" to "ˈθæŋk", "you" to "ˈjuː", "yes" to "ˈjes",
            "no" to "ˈnoʊ", "project" to "ˈprɑːdʒekt", "model" to "ˈmɑːdl", "hybrid" to "ˈhaɪbrɪd",
            "android" to "ˈændrɔɪd", "system" to "ˈsɪstəm", "file" to "ˈfaɪl", "test" to "ˈtest",
            "what" to "wʌt", "is" to "ɪz", "myanmar" to "mjɑːnˌmɑː", "talking" to "ˈtɔːkɪŋ", "about" to "əˈbaʊt"
        )
        
        for ((word, phone) in g2p) {
            raw = raw.replace(Regex("\\b$word\\b"), phone)
        }

        // 🚀 [Advanced Fallback Fix] Dictionary ထဲမပါသော အင်္ဂလိပ်စာလုံးများကို စာလုံးပေါင်းအတိုင်း မထွက်စေဘဲ 
        // VITS English model ရဲ့ VocabMap က အသိအမှတ်ပြုထားတဲ့ IPA အသံထွက်ရင်းမြစ်ပုံစံသို့ အနီးစပ်ဆုံး Mapping လုပ်ပေးခြင်း
        val sb = StringBuilder()
        for (ch in raw) {
            if (vocabMapEn.containsKey(ch)) {
                sb.append(ch)
            } else {
                val replaced = when (ch) {
                    'a' -> 'æ'
                    'e' -> 'ɛ'
                    'i' -> 'ɪ'
                    'o' -> 'ɔ'
                    'u' -> 'ʌ'
                    'c' -> 'k' // 'ç' နေရာမှာ အသံထွက်ပိုမှန်အောင် 'k' ပြောင်းလဲထားပါတယ်
                    'g' -> 'ɡ'
                    'q' -> 'k'
                    'x' -> 's'
                    else -> ' ' 
                }
                if (vocabMapEn.containsKey(replaced)) {
                    sb.append(replaced)
                }
            }
        }
        return sb.toString()
    }

    private fun preProcessMyanmarText(text: String): String {
        var res = text
        res = res.replace(Regex("[xX._*#+=()_\\-]"), "")
        res = res.replace("ဪ", "အော်")
        res = res.replace("၎င်း", "လဂေါင်း")
        res = res.replace("ဖြစ်၏", "ဖြစ်တယ်") 
        res = res.replace("၏", "အီး")
        res = res.replace("၌", "နှိုက်")
        res = res.replace("၍", "ရွေ့")
        res = res.replace("ဤ", "အီ")
        res = res.replace("ဘဏ္ဍာ", "ဘန်ဒါ")
        res = res.replace("သဏ္ဌာန်", "သန်ထန်")
        if (!res.endsWith(" ")) res = "$res   "
        return res
    }

    private fun normalizeNumbers(text: String): String {
        var res = text
        val numMap = mapOf(
            "0" to "သုည", "1" to "တစ်", "2" to "နှစ်", "3" to "သုံး", "4" to "လေး", "5" to "ငါး",
            "6" to "ခြောက်", "7" to "ခုနစ်", "8" to "ရှစ်", "9" to "ကိုး",
            "၀" to "သုည", "၁" to "တစ်", "၂" to "နှစ်", "၃" to "သုံး", "၄" to "လေး", "၅" to "ငါး",
            "၆" to "ခြောက်", "၇" to "ခုနစ်", "၈" to "ရှစ်", "၉" to "ကိုး"
        )
        for ((num, txt) in numMap) { res = res.replace(num, txt) }
        return res
    }

    private fun copyAssetToFile(assetName: String, outFile: File) {
        assets.open(assetName).use { inputStream ->
            FileOutputStream(outFile).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
    }

    private fun saveFloatsToPcm16(file: File, floatData: FloatArray) {
        val fos = FileOutputStream(file)
        val byteBuffer = ByteBuffer.allocate(floatData.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floatData) {
            var s = (f * 32767.0f).toInt()
            if (s > 32767) s = 32767
            if (s < -32768) s = -32768
            byteBuffer.putShort(s.toShort())
        }
        fos.write(byteBuffer.array())
        fos.close()
    }

    private fun convertPcmToAacMuxer(pcmFile: File, aacFile: File, sampleRate: Int) {
        val fis = FileInputStream(pcmFile)
        val muxer = MediaMuxer(aacFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, sampleRate, 1).apply {
            setInteger(MediaFormat.KEY_BIT_RATE, 64000)
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 1024 * 10)
        }
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()

        val bufferInfo = MediaCodec.BufferInfo()
        var audioTrackIndex = -1
        var isMuxerStarted = false
        val rawBuffer = ByteArray(4 * 1024)
        var hasMoreData = true
        var isEOS = false
        var presentationTimeUs = 0L

        while (!isEOS) {
            if (hasMoreData) {
                val inputBufferIndex = codec.dequeueInputBuffer(10000)
                if (inputBufferIndex >= 0) {
                    val inputBuffer = codec.getInputBuffer(inputBufferIndex)!!
                    inputBuffer.clear()
                    val bytesRead = fis.read(rawBuffer)
                    if (bytesRead == -1) {
                        hasMoreData = false
                        codec.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    } else {
                        inputBuffer.put(rawBuffer, 0, bytesRead)
                        codec.queueInputBuffer(inputBufferIndex, 0, bytesRead, presentationTimeUs, 0)
                        presentationTimeUs += (bytesRead / 2) * 1000000L / sampleRate.toLong()
                    }
                }
            }

            var outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 10000)
            if (outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (!isMuxerStarted) {
                    audioTrackIndex = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    isMuxerStarted = true
                }
                outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 0)
            }

            while (outputBufferIndex >= 0) {
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) isEOS = true
                val outputBuffer = codec.getOutputBuffer(outputBufferIndex)!!
                
                if (bufferInfo.size > 0 && (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0)) {
                    if (!isMuxerStarted) {
                        audioTrackIndex = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                        isMuxerStarted = true
                    }
                    outputBuffer.position(bufferInfo.offset)
                    outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                    muxer.writeSampleData(audioTrackIndex, outputBuffer, bufferInfo)
                }
                
                codec.releaseOutputBuffer(outputBufferIndex, false)
                outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 0)
            }
        }
        codec.stop()
        codec.release()
        if (isMuxerStarted) {
            try { muxer.stop() } catch (e: Exception) { e.printStackTrace() }
        }
        muxer.release()
        fis.close()
    }
}
