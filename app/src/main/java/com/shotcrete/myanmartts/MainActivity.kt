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
            setMessage("အသံဖိုင်ပြောင်းလဲနေပါသည်... ခဏစောင့်ပါ...")
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
                                        tokenList.add(1L) // Start token
                                        for (ch in phonemes) { 
                                            tokenList.add(vocabMapEn[ch] ?: 3L) // Map မရှိရင် Space (3L) ပေးမယ်
                                        }
                                        tokenList.add(2L) // End token

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
