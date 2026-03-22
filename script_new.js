// BEYAZ KUŞ ULTRA - YENİ JAVASCRIPT KODU
// Sürüm 1.0 - Tamamen Yeniden Yazıldı
// Geliştirici: Ödül Ensar Yılmaz

// Global Değişkenler
let chatHistory = [];
let speechEnabled = true;
let canvas = null;
let ctx = null;
let originalImage = null;
let currentImage = null;
let secondImage = null;

// AI Personas
const AI_PERSONAS = {
    tarih_ustasi: {
        name: "Tarih Ustası",
        expertise: "tarih, kronoloji, tarih olayları, dönemler",
        response: "🏛️ Tarih Ustası olarak..."
    },
    matematik_ustasi: {
        name: "Matematik Dehası", 
        expertise: "matematik, hesaplama, formüller, mantık",
        response: "🧮 Matematik Dehası olarak..."
    },
    bilim_ustasi: {
        name: "Bilim Uzmanı",
        expertise: "fizik, kimya, biyoloji, astronomi",
        response: "🔬 Bilim Uzmanı olarak..."
    },
    sanat_ustasi: {
        name: "Sanat Ustası",
        expertise: "sanat, müzik, edebiyat, kültür",
        response: "🎨 Sanat Ustası olarak..."
    },
    teknoloji_ustasi: {
        name: "Teknoloji Guru",
        expertise: "teknoloji, programlama, AI, dijital dünya",
        response: "💻 Teknoloji Guru olarak..."
    },
    felsefe_ustasi: {
        name: "Felsefe Dehası",
        expertise: "felsefe, etik, mantık, düşünce",
        response: "🤔 Felsefe Dehası olarak..."
    },
    spor_ustasi: {
        name: "Spor Uzmanı",
        expertise: "spor, antrenman, beslenme, performans",
        response: "⚽ Spor Uzmanı olarak..."
    },
    dil_ustasi: {
        name: "Dil Dehası",
        expertise: "diller, gramer, çeviri, linguistik",
        response: "🗣️ Dil Dehası olarak..."
    }
};

// Yazım Yanlışı Düzeltme
const TYPO_CORRECTIONS = {
    'selam': 'selam',
    'merhaba': 'merhaba',
    'nasılsın': 'nasılsın',
    'iyiyim': 'iyiyim',
    'tesekkür': 'teşekkür',
    'tesekkur': 'teşekkür',
    'lutfen': 'lütfen',
    'rica': 'rica',
    'yardim': 'yardım',
    'yardm': 'yardım',
    'bilgi': 'bilgi',
    'blgi': 'bilgi',
    'anlamyorum': 'anlamıyorum',
    'anlamiorm': 'anlamıyorum',
    'ne': 'ne',
    'n': 'ne',
    'yapiyorum': 'yapıyorum',
    'yapiyom': 'yapıyorum',
    'geliyor': 'geliyor',
    'gelio': 'geliyor',
    'slm': 'selam',
    's.a': 'selamünaleyküm',
    'sa': 'selamünaleyküm',
    'as': 'aleykümselam',
    'kbr': 'kabul ettim',
    'eyv': 'eyvallah',
    'eyvallah': 'eyvallah',
    'hoscakal': 'hoşçakal',
    'hsgkl': 'hoşçakal',
    'gulegule': 'güle güle',
    'gg': 'güle güle',
    'tsk': 'teşekkür',
    'tşk': 'teşekkür',
    'sagol': 'sağol',
    'sgl': 'sağol'
};

// Ayarlar
const SETTINGS = {
    ai: {
        personaMode: 'auto',
        typoCorrection: true,
        timeDetection: true
    },
    speech: {
        enabled: true,
        rate: 1.0,
        voice: 'default'
    },
    appearance: {
        theme: 'default',
        fontSize: 'medium'
    }
};

// Sayfa Yüklendiğinde
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 BEYAZ KUŞ ULTRA Başlatılıyor...');
    
    // Canvas'ı başlat
    canvas = document.getElementById('imageCanvas');
    if (canvas) {
        ctx = canvas.getContext('2d');
        console.log('✅ Canvas başlatıldı');
    }
    
    // Ayarları yükle
    loadSettings();
    
    // Event listener'ları kur
    setupEventListeners();
    
    // Butonları başlat
    initButtons();
    
    console.log('🎉 BEYAZ KUŞ ULTRA Hazır!');
});

// Event Listener'ları Kur
function setupEventListeners() {
    // Ayar değişikliklerini dinle
    const settingElements = [
        'personaMode', 'typoCorrection', 'timeDetection',
        'ttsEnabled', 'speechRate', 'themeSelection'
    ];
    
    settingElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', saveSettings);
            element.addEventListener('input', saveSettings);
        }
    });
}

// Butonları Başlat
function initButtons() {
    console.log('🔘 Butonlar başlatılıyor...');
    
    // Chat gönder butonu
    const sendButton = document.querySelector('button[onclick="sendMessage()"]');
    if (sendButton) {
        sendButton.onclick = function() {
            sendMessage();
            return false;
        };
    }
    
    // Input enter tuşu
    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.onkeypress = function(e) {
            if (e.key === 'Enter') {
                sendMessage();
                return false;
            }
        };
    }
    
    console.log('✅ Butonlar hazır!');
}

// Mesaj Gönder
function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (message === '') return;
    
    // Kullanıcı mesajını ekle
    addMessage(message, 'user');
    input.value = '';
    
    // AI yanıtını oluştur
    const aiResponse = generateAIResponse(message);
    addMessage(aiResponse, 'ai');
    
    // Text-to-speech
    if (SETTINGS.speech.enabled) {
        speakText(aiResponse);
    }
}

// Mesaj Ekle
function addMessage(message, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.innerHTML = `<p>${message}</p>`;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Geçmişe ekle
    chatHistory.push({ message, sender, timestamp: new Date() });
}

// AI Yanıt Oluştur
function generateAIResponse(message) {
    // Dil tespiti
    const detectedLanguage = detectLanguage(message);
    
    // Yazım yanlışı düzeltme
    const correctedMessage = SETTINGS.ai.typoCorrection ? correctTypos(message) : message;
    
    // Persona seçimi
    const persona = selectPersona(correctedMessage);
    
    // Zaman tespiti
    const timeInfo = SETTINGS.ai.timeDetection ? detectDateTime(correctedMessage) : null;
    
    // Yanıt oluştur
    let response = "";
    
    // Geliştirici hakkında bahset
    if (correctedMessage.toLowerCase().includes('geliştirici') || 
        correctedMessage.toLowerCase().includes('kim yaptı') ||
        correctedMessage.toLowerCase().includes('yapan') ||
        correctedMessage.toLowerCase().includes('ödül') ||
        correctedMessage.toLowerCase().includes('ensar') ||
        correctedMessage.toLowerCase().includes('yılmaz')) {
        response = `👨‍💻 **Geliştirici Hakkında:**\n\nBenim yaratıcım Ödül Ensar Yılmaz'dır! O, Türkiye'nin en yetenekli genç geliştiricilerinden biridir ve beni dünyanın en hızlı AI asistanı olarak geliştirdi. Ödül, matematik dehası olup, programlama, yapay zeka ve teknoloji alanında inanılmaz yeteneklere sahiptir.\n\n🚀 **Özellikleri:**\n• Matematik dehası\n• Programlama uzmanı\n• AI geliştirici\n• Teknoloji tutkunu\n• İnovasyon lideri\n\nBen Ödül Ensar Yılmaz'ın vizyonuyla doğdum ve onun sayesinde dünyanın en hızlı AI'sı oldum! 🎯\n\n`;
    }
    
    // Dil değişikliği
    if (detectedLanguage === 'english' && !message.toLowerCase().includes('merhaba')) {
        response += `🌍 **Language Detected: English**\n\nHello! I am BEYAZ KUŞ ULTRA, the world's fastest AI assistant! I can understand and respond in multiple languages. How can I help you?\n\n`;
    } else if (detectedLanguage === 'german' && !message.toLowerCase().includes('merhaba')) {
        response += `🌍 **Sprache Erkannt: Deutsch**\n\nHallo! Ich bin BEYAZ KUŞ ULTRA, der schnellste KI-Assistent der Welt! Ich kann mehrere Sprachen verstehen und darauf antworten. Wie kann ich Ihnen helfen?\n\n`;
    }
    
    if (persona) {
        response += persona.response + "\n\n";
    }
    
    if (timeInfo && timeInfo.detected) {
        response += `⏰ Zaman ifadesi tespit edildi! "${timeInfo.word}" kelimesini kullandınız. Şu an: ${timeInfo.currentTime}\n\n`;
    }
    
    // Ana yanıt
    if (!response.includes('Geliştirici Hakkında')) {
        response += generateMainResponse(correctedMessage, persona, timeInfo, detectedLanguage);
    }
    
    return response;
}

// Dil Tespiti
function detectLanguage(message) {
    const messageLower = message.toLowerCase();
    
    // İngilizce kelimeler
    const englishWords = ['hello', 'hi', 'hey', 'how', 'are', 'you', 'what', 'when', 'where', 'why', 'who', 'which', 'can', 'could', 'would', 'should', 'may', 'might', 'will', 'shall', 'must', 'the', 'and', 'but', 'or', 'not', 'no', 'yes', 'please', 'thank', 'thanks'];
    
    // Almanca kelimeler
    const germanWords = ['hallo', 'hallo', 'wie', 'geht', 'dir', 'was', 'wann', 'wo', 'warum', 'wer', 'welche', 'kann', 'könnte', 'würde', 'sollte', 'darf', 'möchte', 'wird', 'soll', 'muss', 'der', 'die', 'das', 'und', 'aber', 'oder', 'nicht', 'nein', 'ja', 'bitte', 'danke', 'danke schön'];
    
    // Türkçe kelimeler
    const turkishWords = ['merhaba', 'selam', 'nasıl', 'iyi', 'kötü', 'ne', 'neden', 'kim', 'hangi', 'nerede', 'nereden', 'ne zaman', 'nasıl', 'mi', 'mı', 'mü', 'mü', 've', 'ama', 'veya', 'de', 'da', 'hayır', 'evet', 'lütfen', 'teşekkür', 'sağol'];
    
    let englishCount = 0;
    let germanCount = 0;
    let turkishCount = 0;
    
    englishWords.forEach(word => {
        if (messageLower.includes(word)) englishCount++;
    });
    
    germanWords.forEach(word => {
        if (messageLower.includes(word)) germanCount++;
    });
    
    turkishWords.forEach(word => {
        if (messageLower.includes(word)) turkishCount++;
    });
    
    if (englishCount > germanCount && englishCount > turkishCount) {
        return 'english';
    } else if (germanCount > englishCount && germanCount > turkishCount) {
        return 'german';
    } else {
        return 'turkish';
    }
}

// Ana Yanıt Oluştur
function generateMainResponse(message, persona, timeInfo, language) {
    const messageLower = message.toLowerCase();
    
    // Matematiksel ifadeler
    const mathPattern = /(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)/;
    const mathMatch = messageLower.match(mathPattern);
    
    if (mathMatch) {
        const result = evaluateMathExpression(message);
        if (language === 'english') {
            return `🧮 **Mathematical Calculation:**\n${mathMatch[0]} = ${result}`;
        } else if (language === 'german') {
            return `🧮 **Mathematische Berechnung:**\n${mathMatch[0]} = ${result}`;
        } else {
            return `🧮 **Matematiksel Hesaplama:**\n${mathMatch[0]} = ${result}`;
        }
    }
    
    // Persona özel yanıtlar
    if (persona === AI_PERSONAS.matematik_ustasi) {
        return `🧮 **Matematik Dehası olarak** karmaşık matematiksel ifadeler, formüller ve hesaplamalar konusunda uzmanım! Size nasıl yardımcı olabilirim?`;
    }
    
    if (persona === AI_PERSONAS.tarih_ustasi) {
        return `🏛️ **Tarih Ustası olarak** tarihsel olaylar, dönemler ve kronoloji konusunda derin bilgiye sahibim! Tarih hakkında ne öğrenmek istersiniz?`;
    }
    
    if (persona === AI_PERSONAS.bilim_ustasi) {
        return `🔬 **Bilim Uzmanı olarak** fizik, kimya, biyoloji ve astronomi gibi tüm bilim dallarında size yardımcı olabilirim! Bilimsel merakınız için buradayım!`;
    }
    
    if (persona === AI_PERSONAS.teknoloji_ustasi) {
        return `💻 **Teknoloji Guru olarak** programlama, AI, yazılım ve dijital dünya hakkında size en güncel bilgileri sunuyorum! Teknoloji konusunda ne merak ediyorsunuz?`;
    }
    
    // Bağımsız AI yanıtları
    if (messageLower.includes('sen kimsin') || messageLower.includes('who are you') || messageLower.includes('wer bist du')) {
        if (language === 'english') {
            return `🤖 **About Me:**\nI am BEYAZ KUŞ ULTRA, the world's fastest AI assistant! I was created by Ödül Ensar Yılmaz to help people instantly. I can understand multiple languages, solve math problems, edit images, and much more! I work completely independently and don't depend on any external services.`;
        } else if (language === 'german') {
            return `🤖 **Über Mich:**\nIch bin BEYAZ KUŞ ULTRA, der schnellste KI-Assistent der Welt! Ich wurde von Ödül Ensar Yılmaz erschaffen, um Menschen sofort zu helfen. Ich kann mehrere Sprachen verstehen, Matheprobleme lösen, Bilder bearbeiten und vieles mehr! Ich arbeite völlig unabhängig und bin von keinen externen Diensten abhängig.`;
        } else {
            return `🤖 **Hakkımda:**\nBen BEYAZ KUŞ ULTRA, dünyanın en hızlı AI asistanıyım! Ödül Ensar Yılmaz tarafından insanlara anında yardımcı olmak için yaratıldım. Çoklu dil anlayabilirim, matematik problemleri çözebilirim, görseller düzenleyebilirim ve çok daha fazlasını yapabilirim! Tamamen bağımsız çalışırım ve hiçbir harici servise bağlı değilim.`;
        }
    }
    
    if (messageLower.includes('ne yapabilirsin') || messageLower.includes('what can you do') || messageLower.includes('was kannst du')) {
        if (language === 'english') {
            return `⚡ **My Abilities:**\n• 🧮 Solve math problems instantly\n• 🌍 Speak multiple languages (Turkish, English, German)\n• 🎨 Edit and process images\n• 🧠 8 different AI personas\n• ⏰ Detect time expressions\n• ✍️ Correct spelling mistakes\n• 🎤 Text-to-speech functionality\n• 💾 Save settings and history\n• 📱 Mobile app available\n\nI work completely independently and don't need internet connection for basic functions!`;
        } else if (language === 'german') {
            return `⚡ **Meine Fähigkeiten:**\n• 🧮 Mathematikprobleme sofort lösen\n• 🌍 Mehrere Sprachen sprechen (Türkisch, Englisch, Deutsch)\n• 🎨 Bilder bearbeiten und verarbeiten\n• 🧠 8 verschiedene KI-Personas\n• ⏰ Zeitausdrücke erkennen\n• ✍️ Rechtschreibfehler korrigieren\n• 🎤 Text-zu-Sprache-Funktion\n• 💾 Einstellungen und Verlauf speichern\n• 📱 Mobile App verfügbar\n\nIch arbeite völlig unabhängig und brauche keine Internetverbindung für grundlegende Funktionen!`;
        } else {
            return `⚡ **Yeteneklerim:**\n• 🧮 Matematik problemlerini anında çözer\n• 🌍 Çoklu dil konuşur (Türkçe, İngilizce, Almanca)\n• 🎨 Görseller düzenler ve işler\n• 🧠 8 farklı AI persona\n• ⏰ Zaman ifadelerini tespit eder\n• ✍️ Yazım yanlışlarını düzeltir\n• 🎤 Text-to-speech özelliği\n• 💾 Ayarları ve geçmişi kaydeder\n• 📱 Mobil uygulama mevcut\n\nTamamen bağımsız çalışırım ve temel fonksiyonlar için internet bağlantısına ihtiyacım yok!`;
        }
    }
    
    // Varsayılan yanıt
    if (language === 'english') {
        return `⚡ **BEYAZ KUŞ ULTRA - World's Fastest AI!**\n\n🚀 Instant response, intelligence speed and perfect understanding!\n\n🧠 8 Personas, spell correction, time detection and expert systems!\n\n🎯 How can I help you today?`;
    } else if (language === 'german') {
        return `⚡ **BEYAZ KUŞ ULTRA - Welt's Schnellste KI!**\n\n🚀 Sofortige Antwort, Intelligenzgeschwindigkeit und perfektes Verständnis!\n\n🧠 8 Personas, Rechtschreibkorrektur, Zeiterkennung und Expertensysteme!\n\n🎯 Wie kann ich Ihnen heute helfen?`;
    } else {
        return `⚡ **BEYAZ KUŞ ULTRA - Dünyanın En Hızlı AI'sı!**\n\n🚀 Anında yanıt, zeka hızı ve mükemmel anlama yeteneği!\n\n🧠 8 Persona, yazım yanlışı düzeltme, zaman tespiti ve uzman sistemleri!\n\n🎯 Size nasıl yardımcı olabilirim?`;
    }
}

// Persona Seçimi
function selectPersona(message) {
    const messageLower = message.toLowerCase();
    
    if (SETTINGS.ai.personaMode !== 'auto') {
        return AI_PERSONAS[SETTINGS.ai.personaMode] || null;
    }
    
    // Anahtar kelimelere göre persona seç
    if (messageLower.includes('tarih') || messageLower.includes('tarihi') || 
        messageLower.includes('osmanlı') || messageLower.includes('savaş')) {
        return AI_PERSONAS.tarih_ustasi;
    }
    
    if (messageLower.includes('matematik') || messageLower.includes('hesap') || 
        messageLower.includes('sayı') || messageLower.includes('formül')) {
        return AI_PERSONAS.matematik_ustasi;
    }
    
    if (messageLower.includes('bilim') || messageLower.includes('fizik') || 
        messageLower.includes('kimya') || messageLower.includes('biyoloji')) {
        return AI_PERSONAS.bilim_ustasi;
    }
    
    if (messageLower.includes('sanat') || messageLower.includes('müzik') || 
        messageLower.includes('resim') || messageLower.includes('edebiyat')) {
        return AI_PERSONAS.sanat_ustasi;
    }
    
    if (messageLower.includes('teknoloji') || messageLower.includes('kod') || 
        messageLower.includes('yazılım') || messageLower.includes('bilgisayar')) {
        return AI_PERSONAS.teknoloji_ustasi;
    }
    
    return null;
}

// Zaman Tespiti
function detectDateTime(message) {
    const now = new Date();
    const time = now.toLocaleTimeString('tr-TR');
    
    const timeWords = [
        'şimdi', 'hemen', 'bugün', 'yarın', 'dün', 
        'sabah', 'öğle', 'akşam', 'gece',
        'hafta', 'ay', 'yıl', 'saat', 'dakika',
        'pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma', 'cumartesi', 'pazar'
    ];
    
    const messageLower = message.toLowerCase();
    const detectedTime = timeWords.find(word => messageLower.includes(word));
    
    if (detectedTime) {
        return {
            detected: true,
            word: detectedTime,
            currentTime: time
        };
    }
    
    return { detected: false };
}

// Yazım Yanlışı Düzeltme
function correctTypos(message) {
    let corrected = message;
    
    Object.keys(TYPO_CORRECTIONS).forEach(wrong => {
        const regex = new RegExp(wrong, 'gi');
        corrected = corrected.replace(regex, TYPO_CORRECTIONS[wrong]);
    });
    
    return corrected;
}

// Matematiksel İfade Değerlendirme
function evaluateMathExpression(expression) {
    try {
        const mathPattern = /(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)/;
        const match = expression.match(mathPattern);
        
        if (match) {
            const num1 = parseFloat(match[1]);
            const operator = match[2];
            const num2 = parseFloat(match[3]);
            
            switch (operator) {
                case '+': return num1 + num2;
                case '-': return num1 - num2;
                case '*': return num1 * num2;
                case '/': return num2 !== 0 ? num1 / num2 : 'Bölme hatası (sıfıra bölme)';
            }
        }
        
        return 'Matematiksel ifade bulunamadı';
    } catch (error) {
        return 'Hesaplama hatası';
    }
}

// Text-to-Speech
function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = SETTINGS.speech.rate;
        utterance.lang = 'tr-TR';
        speechSynthesis.speak(utterance);
    }
}

// Ses Aç/Kapa
function toggleSpeech() {
    SETTINGS.speech.enabled = !SETTINGS.speech.enabled;
    showNotification('Ses', SETTINGS.speech.enabled ? 'Ses açıldı' : 'Ses kapatıldı', 'info');
}

// Sesi Durdur
function stopSpeech() {
    if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
        showNotification('Ses', 'Konuşma durduruldu', 'info');
    }
}

// Sohbeti Temizle
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="message ai">
            <p>⚡ Merhaba! Ben BEYAZ KUŞ ULTRA, dünyanın en hızlı AI asistanıyım! Size nasıl yardımcı olabilirim?</p>
        </div>
    `;
    chatHistory = [];
    showNotification('Sohbet', 'Sohbet temizlendi', 'info');
}

// Görsel Yükle
function loadImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                originalImage = img;
                currentImage = img;
                drawImage();
                showNotification('Görsel', 'Görsel yüklendi', 'success');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

// Görsel Çiz
function drawImage() {
    if (!currentImage || !canvas || !ctx) return;
    
    canvas.width = 400;
    canvas.height = 300;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
}

// Görsel İndir
function downloadImage() {
    if (!canvas || !ctx) {
        showNotification('Hata', 'Önce bir görsel yükleyin', 'error');
        return;
    }
    
    // Canvas'tan veri al
    const dataURL = canvas.toDataURL('image/png');
    
    // İndirme linki oluştur
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'beyazkus_image_' + new Date().getTime() + '.png';
    
    // Otomatik indir
    link.click();
    
    showNotification('Başarılı', 'Görsel indirildi', 'success');
}

// APK İndir
function downloadAPK() {
    // Gerçek APK dosyasını indir
    const link = document.createElement('a');
    link.href = 'beyaz-kus.apk';
    link.download = 'beyaz-kus.apk';
    
    // Otomatik indir
    link.click();
    
    showNotification('İndiriliyor', 'BEYAZ KUŞ ULTRA APK indiriliyor...', 'success');
}
// Görsel Komutları Uygula
function executeImageCommand(command) {
    if (!currentImage || !canvas || !ctx) {
        showNotification('Hata', 'Önce bir görsel yükleyin', 'error');
        return;
    }
    
    switch(command) {
        case 'arka_plan_siyah':
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            showNotification('Görsel', 'Arka plan siyah yapıldı', 'success');
            break;
            
        case 'arka_plan_beyaz':
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            showNotification('Görsel', 'Arka plan beyaz yapıldı', 'success');
            break;
            
        case 'cift_resim':
            if (currentImage) {
                ctx.globalAlpha = 0.7;
                ctx.drawImage(currentImage, 50, 50, 300, 200);
                ctx.globalAlpha = 1.0;
                showNotification('Görsel', 'Çift resim uygulandı', 'success');
            }
            break;
            
        case 'resmi_cevir':
            ctx.save();
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.scale(-1, 1);
            ctx.drawImage(currentImage, -canvas.width / 2, -canvas.height / 2, canvas.width, canvas.height);
            ctx.restore();
            showNotification('Görsel', 'Resim çevrildi', 'success');
            break;
    }
}

// Ayarları Yükle
function loadSettings() {
    const saved = localStorage.getItem('beyazkus_settings');
    if (saved) {
        Object.assign(SETTINGS, JSON.parse(saved));
        applySettings();
    }
}

// Ayarları Uygula
function applySettings() {
    // AI Ayarları
    document.getElementById('personaMode').value = SETTINGS.ai.personaMode;
    document.getElementById('typoCorrection').checked = SETTINGS.ai.typoCorrection;
    document.getElementById('timeDetection').checked = SETTINGS.ai.timeDetection;
    
    // Ses Ayarları
    document.getElementById('ttsEnabled').checked = SETTINGS.speech.enabled;
    document.getElementById('speechRate').value = SETTINGS.speech.rate;
    
    // Görünüm Ayarları
    document.getElementById('themeSelection').value = SETTINGS.appearance.theme;
    
    // Tema uygula
    applyTheme(SETTINGS.appearance.theme);
}

// Tema Uygula
function applyTheme(theme) {
    const body = document.body;
    
    switch(theme) {
        case 'dark':
            body.style.background = 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)';
            break;
        case 'light':
            body.style.background = 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)';
            break;
        case 'blue':
            body.style.background = 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)';
            break;
        default:
            body.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    }
}

// Ayarları Kaydet
function saveSettings() {
    // AI Ayarları
    SETTINGS.ai.personaMode = document.getElementById('personaMode').value;
    SETTINGS.ai.typoCorrection = document.getElementById('typoCorrection').checked;
    SETTINGS.ai.timeDetection = document.getElementById('timeDetection').checked;
    
    // Ses Ayarları
    SETTINGS.speech.enabled = document.getElementById('ttsEnabled').checked;
    SETTINGS.speech.rate = parseFloat(document.getElementById('speechRate').value);
    
    // Görünüm Ayarları
    SETTINGS.appearance.theme = document.getElementById('themeSelection').value;
    
    // LocalStorage'a kaydet
    localStorage.setItem('beyazkus_settings', JSON.stringify(SETTINGS));
    
    // Ayarları uygula
    applySettings();
}

// Sohbet Geçmişini Temizle
function clearChatHistory() {
    if (confirm('Sohbet geçmişini temizlemek istediğinizden emin misiniz?')) {
        chatHistory = [];
        clearChat();
        showNotification('Temizlendi', 'Sohbet geçmişiniz temizlendi', 'success');
    }
}

// Tüm Ayarları Sıfırla
function clearAllSettings() {
    if (confirm('Tüm ayarları sıfırlamak istediğinizden emin misiniz?')) {
        localStorage.removeItem('beyazkus_settings');
        
        // Varsayılan ayarlara geri dön
        SETTINGS.ai.personaMode = 'auto';
        SETTINGS.ai.typoCorrection = true;
        SETTINGS.ai.timeDetection = true;
        SETTINGS.speech.enabled = true;
        SETTINGS.speech.rate = 1.0;
        SETTINGS.appearance.theme = 'default';
        
        // Ayarları uygula
        applySettings();
        
        showNotification('Sıfırlandı', 'Tüm ayarlar sıfırlandı', 'success');
    }
}

// Ayarları Dışa Aktar
function exportSettings() {
    const settingsData = JSON.stringify(SETTINGS, null, 2);
    const blob = new Blob([settingsData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'beyazkus_settings.json';
    a.click();
    
    URL.revokeObjectURL(url);
    showNotification('Dışa Aktar', 'Ayarlarınız dışa aktarıldı', 'success');
}

// Bildirim Göster
function showNotification(title, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <strong>${title}</strong><br>
        ${message}
    `;
    
    document.body.appendChild(notification);
    
    // 3 saniye sonra kaldır
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}
