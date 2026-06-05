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

    // 🚀 [Updated Vocab Map] English ONNX (Piper/VITS) အသုံးအများဆုံး အင်္ဂလိပ်စာလုံးနှင့် IPA သင်္ကေတများ ဇယားအပြည့်စုံ
    private val vocabMapEn = mapOf(
        '_' to 0L, '^' to 1L, '$' to 2L, ' ' to 3L, '!' to 4L, '\'' to 5L, ',' to 6L, '-' to 7L,
        '.' to 8L, ';' to 9L, '?' to 10L, 'a' to 11L, 'b' to 12L, 'c' to 13L, 'd' to 14L, 'e' to 15L,
        'f' to 16L, 'g' to 17L, 'h' to 18L, 'i' to 19L, 'j' to 20L, 'k' to 21L, 'l' to 22L, 'm' to 23L,
        'n' to 24L, 'o' to 25L, 'p' to 26L, 'q' to 27L, 'r' to 28L, 's' to 29L, 't' to 30L, 'u' to 31L,
        'v' to 32L, 'w' to 33L, 'x' to 34L, 'y' to 35L, 'z' to 36L,
        // IPA Symbols Fallback IDs
        'æ' to 37L, 'ç' to 38L, 'ð' to 39L, 'ø' to 40L, 'ŋ' to 41L, 'ɐ' to 42L, 'ɒ' to 43L, 'ɔ' to 44L,
        'ə' to 45L, 'ɛ' to 46L, 'ɜ' to 47L, 'ɪ' to 48L, 'ɫ' to 49L, 'ɱ' to 50L, 'œ' to 51L,
        'ʃ' to 52L, 'θ' to 53L, 'ʊ' to 54L, 'ʌ' to 55L, 'ʒ' to 56L, 'ː' to 57L, 'ˈ' to 58L, 'ˌ' to 59L
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
                        Toast.makeText(this@MainActivity, "English Model Load Fail!", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        speakButton.setOnClickListener {
            val rawText = inputText.text.toString().trim()
            if (rawText.isNotEmpty()) {
                val env = ortEnv
                val currentSessionEn = ortSessionEn
                
                if (currentSessionEn == null || env == null) {
                    Toast.makeText(this, "Engine not ready!", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                progressDialog?.show()

                thread(start = true) {
                    try {
                        // 🚀 [CRITICAL FIX] စာသားကို စနစ် ၂ မျိုးလုံးကိုက်အောင် ပြင်ဆင်ခြင်း
                        val processedText = prepareInputText(rawText)
                        
                        val tokenList = mutableListOf<Long>()
                        tokenList.add(1L) // BOS Token (^ သို့မဟုတ် 1)
                        
                        // စာလုံးကြားထဲမှာ ပိတ်မသွားအောင် _ (Pad Token - 0L) ခံပြီး ထည့်သွင်းပေးခြင်း (Standard VITS Format)
                        for (ch in processedText) {
                            val id = vocabMapEn[ch] ?: 3L // မရှိရင် Space ID ပေးမယ်
                            tokenList.add(id)
                            tokenList.add(0L) // Interleaved Pad
                        }
                        tokenList.add(2L) // EOS Token ($ သို့မဟုတ် 2)

                        val inputSequence = tokenList.toLongArray()
                        val seqLength = inputSequence.size

                        val inputShape = longArrayOf(1, seqLength.toLong())
                        val lengthShape = longArrayOf(1)

                        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
                        val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(seqLength.toLong())), lengthShape)
                        val scalesTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f)), longArrayOf(3))
                        
                        val attentionMaskSequence = LongArray(seqLength) { 1L }
                        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(attentionMaskSequence), inputShape)
                        val sidTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(0L)), lengthShape)

                        val inputMap = HashMap<String, OnnxTensor>()
                        currentSessionEn.inputNames?.forEach { name ->
                            val lowerName = name.lowercase()
                            when {
                                lowerName.contains("mask") || lowerName.contains("attention") -> inputMap[name] = maskTensor
                                lowerName.contains("length") -> inputMap[name] = lengthTensor
                                lowerName.contains("scale") -> inputMap[name] = scalesTensor
                                lowerName.contains("sid") || lowerName.contains("speaker") -> inputMap[name] = sidTensor
                                lowerName.contains("input_ids") || lowerName.contains("input") || lowerName == "text" -> inputMap[name] = inputTensor
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
                                Toast.makeText(this@MainActivity, "Output empty!", Toast.LENGTH_SHORT).show()
                            }
                            return@thread
                        }

                        // Normalization
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
                            Toast.makeText(this@MainActivity, "Success!", Toast.LENGTH_SHORT).show()
                        }

                        // Playback
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
                    e.printStackTrace()
                }
            }
        }
    }

    // 💡 Text ကို Model အကြိုက် ပုံစံပြောင်းလဲပေးမယ့် Function
    private fun prepareInputText(text: String): String {
        var raw = text.lowercase(Locale.ROOT).trim()
        
        // သတ်မှတ်ထားတဲ့ စာလုံးတွေကို IPA Phonemes အဖြစ် အရင်ပြောင်းကြည့်မယ်
        val g2p = mapOf(
            "hello" to "həˈloʊ", "hi" to "ˈhaɪ", "ok" to "oʊˈkeɪ", "good" to "ˈɡʊd",
            "morning" to "ˈmɔːrnɪŋ", "thank" to "ˈθæŋk", "you" to "ˈjuː"
        )
        
        var parsed = raw
        for ((word, phone) in g2p) {
            if (raw.contains(word)) {
                parsed = raw.replace(Regex("\\b$word\\b"), phone)
                return parsed // Phonemes mapping မိသွားရင် တိုက်ရိုက် သုံးမယ်
            }
        }
        
        // အကယ်၍ အပေါ်က စာလုံးတွေနဲ့ မကိုက်ညီရင် ရိုးရိုး အင်္ဂလိပ်စာလုံး အက္ခရာတွေကိုပဲ တိုက်ရိုက် Mapping ယူစေမယ် (Fallback)
        return raw
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
