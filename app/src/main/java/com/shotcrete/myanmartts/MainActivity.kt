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

    // အင်္ဂလိပ် Vocab Map (Piper Ryan Low)
    private val vocabMapEn = mapOf(
        '_' to 0L, '^' to 1L, '$' to 2L, ' ' to 3L, '!' to 4L, '\'' to 5L, ',' to 6L, '-' to 7L,
        '.' to 8L, ';' to 9L, '?' to 10L, 'a' to 11L, 'b' to 12L, 'd' to 13L, 'e' to 14L, 'f' to 15L,
        'h' to 16L, 'i' to 17L, 'j' to 18L, 'k' to 19L, 'l' to 20L, 'm' to 21L, 'n' to 22L, 'o' to 23L,
        'p' to 24L, 'r' to 25L, 's' to 26L, 't' to 27L, 'u' to 28L, 'v' to 29, 'w' to 30L, 'z' to 31L,
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
            setMessage("အသံဖိုင်ပြောင်းလဲနေပါသည်... ခဏစောင့်ပါ...")
            setCancelable(false)
        }

        // 🚀 လုံးဝ Offline မော်ဒယ်နှစ်ခုလုံးအား Assets မှ တိုက်ရိုက်နှိုးခြင်း
        thread(start = true) {
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                
                // ၁။ မြန်မာမော်ဒယ်အား Assets မှ ယူခြင်း
                val modelFileMm = File(cacheDir, "model.onnx")
                if (!modelFileMm.exists()) copyAssetToFile("model.onnx", modelFileMm)
                ortSessionMm = ortEnv?.createSession(modelFileMm.absolutePath)

                // ၂။ အင်္ဂလိပ်မော်ဒယ်အား Assets မှ ယူခြင်း
                val modelFileEn = File(cacheDir, "en_model.onnx")
                if (!modelFileEn.exists()) copyAssetToFile("en_model.onnx", modelFileEn)
                ortSessionEn = ortEnv?.createSession(modelFileEn.absolutePath)

                runOnUiThread {
                    Toast.makeText(this@MainActivity, "မြန်မာ + အင်္ဂလိပ် Offline Engine အဆင်သင့်ဖြစ်ပါပြီဗျာ", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Engine Loading Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        speakButton.setOnClickListener {
            val rawText = inputText.text.toString().trim()
            if (rawText.isNotEmpty()) {
                val env = ortEnv
                if (ortSessionMm == null || ortSessionEn == null || env == null) {
                    Toast.makeText(this, "Engines များ အဆင်သင့်မဖြစ်သေးပါ...", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                progressDialog?.show()

                thread(start = true) {
                    try {
                        val textSegments = splitByLanguage(rawText)
                        val combinedAudioList = mutableListOf<FloatArray>()

                        for (segment in textSegments) {
                            if (segment.isEnglish) {
                                // 🇬🇧 English Engine Pipeline
                                val phonemes = textToPhonemes(segment.text)
                                val tokenList = mutableListOf<Long>()
                                tokenList.add(1L) // ^
                                for (ch in phonemes) {
                                    tokenList.add(vocabMapEn[ch] ?: 3L) 
                                }
                                tokenList.add(2L) // $

                                val inputSequence = tokenList.toLongArray()
                                if (inputSequence.size < 3) continue

                                val inputShape = longArrayOf(1, inputSequence.size.toLong())

                                // Buffer များ တည်ဆောက်ခြင်း
                                val longBufferInput = java.nio.LongBuffer.wrap(inputSequence)
                                val longBufferLength = java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong()))

                                // 💡 Missing attention_mask Error အား အလိုအလျောက် Tensor တည်ဆောက်၍ ဖြေရှင်းခြင်း
                                val attentionMaskSequence = LongArray(inputSequence.size) { 1L }
                                val longBufferMask = java.nio.LongBuffer.wrap(attentionMaskSequence)

                                val inputTensor = OnnxTensor.createTensor(env, longBufferInput, inputShape)
                                val lengthTensor = OnnxTensor.createTensor(env, longBufferLength, longArrayOf(1))
                                val maskTensor = OnnxTensor.createTensor(env, longBufferMask, inputShape)
                                val scalesTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f)), longArrayOf(3))

                                val inputMap = HashMap<String, OnnxTensor>()
                                ortSessionEn?.inputNames?.forEach { name ->
                                    when {
                                        name == "input" || name == "input_ids" || name.contains("input_ids") -> inputMap[name] = inputTensor
                                        name == "attention_mask" || name.contains("attention_mask") -> inputMap[name] = maskTensor
                                        name.contains("input_lengths") || name.contains("lengths") -> inputMap[name] = lengthTensor
                                        name.contains("scales") || name.contains("scale") -> inputMap[name] = scalesTensor
                                    }
                                }

                                val results = ortSessionEn?.run(inputMap)
                                val outputTensor = results?.get(0) as? OnnxTensor
                                outputTensor?.let {
                                    val floatBuffer = it.floatBuffer
                                    val audioFloats = FloatArray(floatBuffer.remaining())
                                    floatBuffer.get(audioFloats)
                                    combinedAudioList.add(audioFloats)
                                }
                                inputTensor.close()
                                lengthTensor.close()
                                maskTensor.close()
                                scalesTensor.close()
                                results?.close()

                            } else {
                                // 🇲🇲 Myanmar Engine Pipeline
                                var processedText = preProcessMyanmarText(segment.text)
                                processedText = normalizeNumbers(processedText)

                                val cleanChunk = processedText.replace(" ", "")
                                val validChars = cleanChunk.filter { vocabMapMm.containsKey(it) }
                                if (validChars.length < 2) continue

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
                                if (inputSequence.isEmpty()) continue

                                val inputShape = longArrayOf(1, inputSequence.size.toLong())
                                val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                                val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), longArrayOf(1))
                                val scalesTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f)), longArrayOf(3))

                                val inputMap = HashMap<String, OnnxTensor>()
                                ortSessionMm?.inputNames?.forEach { name ->
                                    when {
                                        name == "input" || name.contains("input_ids") -> inputMap[name] = inputTensor
                                        name.contains("input_lengths") || name.contains("lengths") -> inputMap[name] = lengthTensor
                                        name.contains("scales") || name.contains("scale") -> inputMap[name] = scalesTensor
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
                                results?.close()
                            }
                        }

                        if (combinedAudioList.isEmpty()) {
                            runOnUiThread { progressDialog?.dismiss() }
                            return@thread
                        }

                        // Audio Stitching Core
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
                            Toast.makeText(this@MainActivity, "Hybrid အသံဖိုင် ထွက်လာပါပြီခင်ဗျာ", Toast.LENGTH_LONG).show()
                        }

                        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
                        val audioTrack = AudioTrack(AudioManager.STREAM_MUSIC, sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT, bufferSize, AudioTrack.MODE_STREAM)
                        audioTrack.play()
                        audioTrack.write(finalAudioFloats, 0, finalAudioFloats.size, AudioTrack.WRITE_BLOCKING)

                    } catch (e: Exception) {
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

    data class LangSegment(val text: String, val isEnglish: Boolean)
    private fun splitByLanguage(text: String): List<LangSegment> {
        val segments = mutableListOf<LangSegment>()
        val regex = Regex("([a-zA-Z\\s!,.-]+)|([^a-zA-Z]+)")
        val matches = regex.findAll(text)
        
        for (match in matches) {
            val segmentText = match.value
            if (segmentText.trim().isNotEmpty()) {
                val isEng = segmentText.any { it in 'a'..'z' || it in 'A'..'Z' }
                segments.add(LangSegment(segmentText, isEng))
            }
        }
        return segments
    }

    private fun textToPhonemes(text: String): String {
        var raw = text.lowercase()
        val g2p = mapOf(
            "hello" to "həˈloʊ", "hi" to "ˈhaɪ", "ok" to "oʊˈkeɪ", "good" to "ˈɡʊd",
            "morning" to "ˈmɔːrnɪŋ", "thank" to "ˈθæŋk", "you" to "ˈjuː", "yes" to "ˈjes",
            "no" to "ˈnoʊ", "project" to "ˈprɑːdʒekt", "model" to "ˈmɑːdl", "hybrid" to "ˈhaɪbrɪd",
            "android" to "ˈændrɔɪd", "system" to "ˈsɪstəm", "file" to "ˈfaɪl", "test" to "ˈtest"
        )
        for ((word, phone) in g2p) {
            raw = raw.replace(Regex("\\b$word\\b"), phone)
        }
        return raw.replace("a", "æ").replace("e", "ɛ").replace("i", "ɪ")
                  .replace("o", "ɔ").replace("u", "ʌ").replace("sh", "ʃ")
                  .replace("th", "θ").replace("ch", "ʃ")
    }

    private fun preProcessMyanmarText(text: String): String {
        var res = text
        res = res.replace(Regex("[xX._\\-*#+=()_]"), "")
        res = res.replace("ဪ", "အော်")
        res = res.replace("၎င်း", "လဂေါင်း")
        res = res.replace("ဖြစ်၏", "ဖြစ်တယ်") 
        res = res.replace("၏", "အီး")
        res = res.replace("၌", "နှိုက်")
        res = res.replace("၍", "ရွေ့")
        res = res.replace("ဤ", "အီ")
        res = res.replace("ဘဏ္ဍာ", "ဘန်ဒါ")
        res = res.replace("သဏ္ဌာန်", "သန်ထန်")
        res = res.replace("ဥက္ကာမြဲ", "အုတ်ကာမြဲ")
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
            while (outputBufferIndex >= 0) {
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) isEOS = true
                val outputBuffer = codec.getOutputBuffer(outputBufferIndex)!!
                if (bufferInfo.size > 0 && (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0)) {
                    outputBuffer.position(bufferInfo.offset)
                    outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                    muxer.writeSampleData(
