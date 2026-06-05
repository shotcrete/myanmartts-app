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
    private var ortSessionEn: OrtSession? = null
    private var progressDialog: ProgressDialog? = null
    
    private var lastAudioFilePath: String? = null
    private var mediaPlayer: MediaPlayer? = null

    // VITS English Model အတွက် Standard Vocab Map (အပိတ်၊ အဖွင့် သင်္ကေတများ အပါအဝင်)
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
            setMessage("English Voice Generating... Please wait...")
            setCancelable(false)
        }

        // English ONNX Model တစ်ခုတည်းကိုသာ Load ပြုလုပ်ခြင်း
        thread(start = true) {
            try {
                ortEnv = OrtEnvironment.getEnvironment()

                val modelFileEn = File(cacheDir, "en_model.onnx")
                if (!modelFileEn.exists() || modelFileEn.length() < 10_000_000) {
                    if (modelFileEn.exists()) modelFileEn.delete()
                    copyAssetToFile("en_model.onnx", modelFileEn)
                }
                
                if (modelFileEn.exists() && modelFileEn.length() > 10_000_000) {
                    ortSessionEn = ortEnv?.createSession(modelFileEn.absolutePath)
                }

                runOnUiThread {
                    if (ortSessionEn != null) {
                        Toast.makeText(this@MainActivity, "English Engine Ready!", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this@MainActivity, "English Model Fail to Load!", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Loading Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        speakButton.setOnClickListener {
            val rawText = inputText.text.toString().trim()
            if (rawText.isNotEmpty()) {
                val env = ortEnv
                val currentSessionEn = ortSessionEn
                
                if (currentSessionEn == null || env == null) {
                    Toast.makeText(this, "English Engine not ready yet!", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                progressDialog?.show()

                thread(start = true) {
                    try {
                        // English စာသား သန့်စင်ပြီး Phonemes ပြောင်းလဲခြင်း
                        val phonemes = textToPhonemes(rawText)
                        
                        // Token ID တည်ဆောက်ခြင်း
                        val tokenList = mutableListOf<Long>()
                        tokenList.add(1L) // Start Token (^ သို့မဟုတ် Bos)
                        
                        for (ch in phonemes) { 
                            // 🚀 ဇယားထဲမှာ မရှိခဲ့ရင် Space Token (3L) အစားထိုးပြီး အသံထွက်မပျက်အောင် ထိန်းထားခြင်း
                            val tokenId = vocabMapEn[ch] ?: 3L 
                            tokenList.add(tokenId)
                        }
                        tokenList.add(2L) // End Token ($ သို့မဟုတ် Eos)

                        val inputSequence = tokenList.toLongArray()
                        if (inputSequence.size < 3) {
                            runOnUiThread { progressDialog?.dismiss() }
                            return@thread
                        }

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
                        
                        var audioFloats: FloatArray? = null
                        outputTensor?.let {
                            val floatBuffer = it.floatBuffer
                            audioFloats = FloatArray(floatBuffer.remaining())
                            floatBuffer.get(audioFloats)
                        }

                        inputTensor.close()
                        lengthTensor.close()
                        scalesTensor.close()
                        maskTensor.close()
                        sidTensor.close()
                        results?.close()

                        val finalAudioFloats = audioFloats
                        if (finalAudioFloats == null || finalAudioFloats.isEmpty()) {
                            runOnUiThread {
                                progressDialog?.dismiss()
                                Toast.makeText(this@MainActivity, "Model output is empty!", Toast.LENGTH_SHORT).show()
                            }
                            return@thread
                        }

                        // Audio Volume Normalization
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

                        val outputAacFile = File(myanmarTtsDir, "ENGLISH_TTS_${System.currentTimeMillis()}.m4a")
                        convertPcmToAacMuxer(tempPcmFile, outputAacFile, sampleRate)
                        tempPcmFile.delete()

                        lastAudioFilePath = outputAacFile.absolutePath

                        runOnUiThread {
                            progressDialog?.dismiss()
                            Toast.makeText(this@MainActivity, "English Voice Generated Successfully!", Toast.LENGTH_LONG).show()
                        }

                        // Playback via Static AudioTrack
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
                Toast.makeText(this, "Please enter English text first", Toast.LENGTH_SHORT).show()
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

    // 💡 [အင်္ဂလိပ် သီးသန့် Phonemes စနစ်] ONNX Model အကြိုက်ဆုံး ပုံစံဖြင့် တိုက်ရိုက် Text-to-IPA ပြောင်းလဲခြင်း
    private fun textToPhonemes(text: String): String {
        var raw = text.lowercase(Locale.ROOT).trim()
        
        // အခြေခံ အသုံးအများဆုံး စာလုံးများအတွက် တိကျသော VITS Phonemes IPA Dictionary
        val g2p = mapOf(
            "hello" to "həˈloʊ", 
            "hi" to "ˈhaɪ", 
            "ok" to "oʊˈkeɪ", 
            "good" to "ˈɡʊd",
            "morning" to "ˈmɔːrnɪŋ", 
            "thank" to "ˈθæŋk", 
            "you" to "ˈjuː", 
            "yes" to "ˈjes",
            "no" to "ˈnoʊ", 
            "what" to "wʌt", 
            "is" to "ɪz", 
            "myanmar" to "mjɑːnˌmɑː",
            "project" to "ˈprɑːdʒekt", 
            "model" to "ˈmɑːdl", 
            "hybrid" to "ˈhaɪbrɪd",
            "android" to "ˈændrɔɪd", 
            "system" to "ˈsɪstəm", 
            "file" to "ˈfaɪl", 
            "test" to "ˈtest"
        )
        
        for ((word, phone) in g2p) {
            raw = raw.replace(Regex("\\b$word\\b"), phone)
        }

        // Dictionary ထဲမပါသော အက္ခရာများ ကျန်ခဲ့ပါက Vocab Map တွင် ပါဝင်သော IPA သံများသို့ တိုက်ရိုက် mapping ချပေးခြင်း
        val sb = StringBuilder()
        for (ch in raw) {
            if (vocabMapEn.containsKey(ch)) {
                sb.append(ch)
            } else {
                // Vocab အသိအမှတ်ပြုသော Standard IPA စာလုံးများသို့ အနီးစပ်ဆုံး ပြောင်းလဲပေးခြင်း
                val replaced = when (ch) {
                    'a' -> 'æ'
                    'e' -> 'ɛ'
                    'i' -> 'ɪ'
                    'o' -> 'ɔ'
                    'u' -> 'ʌ'
                    'c' -> 'ç'
                    'g' -> 'ɡ'
                    else -> ' ' // မသိသော စာလုံးများကို ပိတ်မပစ်ဘဲ space ခြားပေးခြင်း
                }
                sb.append(replaced)
            }
        }
        return sb.toString()
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

    override fun onDestroy() {
        super.onDestroy()
        mediaPlayer?.release()
        progressDialog?.dismiss()
        ortSessionEn?.close()
        ortEnv?.close()
    }
}
