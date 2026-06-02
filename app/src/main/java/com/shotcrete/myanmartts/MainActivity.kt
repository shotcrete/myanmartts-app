package com.shotcrete.myanmartts

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File

class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val inputText = findViewById<EditText>(R.id.inputText)
        val speakButton = findViewById<Button>(R.id.speakButton)

        // ONNX Runtime အား Engine အနေဖြင့် နောက်ကွယ်တွင် စတင်နှိုးခြင်း
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("model.onnx").readBytes()
            ortSession = ortEnv?.createSession(modelBytes)
            Toast.makeText(this, "TTS Engine အဆင်သင့်ဖြစ်ပါပြီ", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Engine ဖွင့်ရတာ မအောင်မြင်ပါ: ${e.message}", Toast.LENGTH_LONG).show()
        }

        speakButton.setOnClickListener {
            val text = inputText.text.toString().strip()
            if (text.isNotEmpty()) {
                Toast.makeText(this, "အသံပြောင်းလဲနေပါပြီ...", Toast.LENGTH_SHORT).show()
                // ဒီနေရာမှာ နောက်တစ်ဆင့် အသံဖိုင် Array ပြောင်းမယ့် Logic ကို GitHub Actions ကနေ တန်းထုတ်ပါမယ်
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
