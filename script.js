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

// Başlatma fonksiyonu - Güncellenmiş
function init() {
    console.log('🚀 BEYAZ KUŞ ULTRA başlatılıyor...');
    
    // Kullanıcı giriş kontrolü
    const user = localStorage.getItem('user');
    if (user) {
        // Kullanıcı giriş yapmış, ana uygulamayı göster
        document.getElementById('loginContainer').style.display = 'none';
        document.getElementById('mainApp').style.display = 'block';
        initializeChat();
    } else {
        // Kullanıcı giriş yapmamış, login ekranını göster
        document.getElementById('loginContainer').style.display = 'flex';
        document.getElementById('mainApp').style.display = 'none';
    }
    
    console.log('✅ BEYAZ KUŞ ULTRA hazır!');
}

// Sayfa yüklendiğinde başlat
document.addEventListener('DOMContentLoaded', init);

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

// Hızlı Mesaj Fonksiyonu
function quickMessage(message) {
    document.getElementById('userInput').value = message;
    sendMessage();
}

// Butonları Başlat
function initButtons() {
    console.log('🔘 Butonlar başlatılıyor...');
    
    // Chat gönder butonu
    const sendButton = document.getElementById('sendButton');
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
    
    if (!message) return;
    
    // Dil tespiti
    const detectedLanguage = detectLanguage(message);
    
    // Kullanıcı mesajını ekle
    addMessage(message, 'user');
    input.value = '';
    
    // AI yanıtını oluştur
    const aiResponse = generateAIResponse(message);
    addMessage(aiResponse, 'ai');
    
    // Text-to-speech - detectedLanguage parametresi ile
    if (SETTINGS.speech.enabled) {
        speakText(aiResponse, detectedLanguage);
    }
}

// Google OAuth ile Giriş
function handleGoogleSignIn() {
    // Google OAuth 2.0 ile giriş yap
    const clientId = 'YOUR_GOOGLE_CLIENT_ID'; // Buraya gerçek client ID girilecek
    const redirectUri = window.location.origin + '/callback';
    const scope = 'email profile';
    
    // OAuth URL oluştur
    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
        `client_id=${clientId}&` +
        `redirect_uri=${encodeURIComponent(redirectUri)}&` +
        `scope=${encodeURIComponent(scope)}&` +
        `response_type=token&` +
        `state=${Math.random().toString(36).substring(7)}`;
    
    // Demo için simüle edilmiş giriş
    simulateGoogleSignIn();
}

// Demo Google Giriş Simülasyonu
function simulateGoogleSignIn() {
    // Loading göster
    showNotification('Giriş', 'Google ile giriş yapılıyor...', 'info');
    
    // Simüle edilmiş başarılı giriş
    setTimeout(() => {
        const userData = {
            name: 'Demo User',
            email: 'demo@gmail.com',
            picture: 'https://picsum.photos/seed/user/100/100.jpg'
        };
        
        // Kullanıcı bilgisini kaydet
        localStorage.setItem('user', JSON.stringify(userData));
        
        // Login ekranını gizle, ana uygulamayı göster
        document.getElementById('loginContainer').style.display = 'none';
        document.getElementById('mainApp').style.display = 'block';
        
        showNotification('Hoş Geldin', `${userData.name} olarak giriş yapıldı!`, 'success');
        
        // Chat'i başlat
        initializeChat();
    }, 1500);
}

// Çıkış yap
function handleSignOut() {
    localStorage.removeItem('user');
    document.getElementById('loginContainer').style.display = 'flex';
    document.getElementById('mainApp').style.display = 'none';
    showNotification('Çıkış', 'Başarıyla çıkış yapıldı', 'info');
}

// Chat'i başlat
function initializeChat() {
    // Chat geçmişini yükle
    loadChatHistory();
    
    // Butonları başlat
    initButtons();
}

// Mesaj Ekle - Güncellenmiş Baloncuk Stil
function addMessage(message, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const now = new Date();
    const time = now.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
    
    // Avatar ve baloncuk oluştur
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'user' ? '👤' : '🤖';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = message;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = time;
    
    bubble.appendChild(content);
    bubble.appendChild(timeDiv);
    
    // Mesaj yapısını oluştur
    if (sender === 'user') {
        // Kullanıcı mesajı: sağ tarafta
        messageDiv.appendChild(bubble);
        messageDiv.appendChild(avatar);
    } else {
        // AI mesajı: sol tarafta
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Geçmişe ekle
    chatHistory.push({ message, sender, timestamp: now });
}

// Mesaj Gönder - Güncellenmiş
function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Dil tespiti
    const detectedLanguage = detectLanguage(message);
    
    // Kullanıcı mesajını ekle
    addMessage(message, 'user');
    input.value = '';
    
    // AI yanıtını oluştur
    const aiResponse = generateAIResponse(message);
    addMessage(aiResponse, 'ai');
    
    // Text-to-speech - kontrol et
    if (speechEnabled) {
        speakText(aiResponse, detectedLanguage);
    }
}

// Hızlı Mesaj Fonksiyonu
function quickMessage(message) {
    document.getElementById('userInput').value = message;
    sendMessage();
}

// Bildirim göster - Güncellenmiş
function showNotification(title, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        z-index: 1000;
        max-width: 300px;
        animation: slideIn 0.3s ease-out;
    `;
    
    notification.innerHTML = `
        <div style="font-weight: bold; color: #333; margin-bottom: 0.5rem;">${title}</div>
        <div style="color: #666; font-size: 0.9rem;">${message}</div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// CSS animasyonları ekle
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Türkiye ve Dünya Bilgisi Ekle
function addTurkeyWorldKnowledge(response, language) {
    const turkeyWorldInfo = {
        turkish: {
            intro: "🇹🇷 TÜRKİYE VE DÜNYA UZMANI:\n\n",
            knowledge: [
                "🏛️ Türkiye: 81 il, 7 coğrafi bölge, 16.000+ yıllık tarih",
                "🌍 Dünya: 195 ülke, 7 kıta, 8 milyar+ nüfus",
                "📊 Türkiye'nin en büyük şehirleri: İstanbul, Ankara, İzmir, Bursa",
                "🏔️ Türkiye'nin en yüksek dağı: Ağrı Dağı (5.137m)",
                "🌊 Türkiye'nin en uzun nehri: Kızılırmak (1.355km)",
                "🏛️ Türkiye'nin ilk başkenti: Ankara (1923)",
                "🕌 Türkiye'deki cami sayısı: 80.000+",
                "🏰 Türkiye'deki kale sayısı: 500+",
                "🏖️ Türkiye'nin sahil uzunluğu: 8.333km",
                "🌍 Dünya'nın en büyük ülkesi: Rusya (17.1 milyon km²)",
                "🏔️ Dünya'nın en yüksek dağı: Everest (8.848m)",
                "🌊 Dünya'nın en uzun nehri: Nil (6.650km)",
                "🏙️ Dünya'nın en kalabalık şehri: Tokyo (37 milyon)",
                "🌡️ Dünya'nın en sıcak yeri: Death Valley (56.7°C)",
                "❄️ Dünya'nın en soğuk yeri: Antarktika (-89.2°C)",
                "🌍 Türkiye'nin komşuları: 8 ülke (Yunanistan, Bulgaristan, Gürcistan, Ermenistan, Nahçıvan, İran, Irak, Suriye)",
                "🏛️ Türkiye'deki UNESCO Dünya Mirasları: 19 (Efes, Kapadokya, Pamukkale, Troya vb.)",
                "🌍 Dünya'daki 7 harika: Mısır Piramitleri, Babil Asma Bahçeleri, Artemis Tapınağı, Zeus Heykeli, Mausoleum, Rodos Heykeli, İskenderiye Feneri"
            ]
        },
        english: {
            intro: "🌍 TURKEY & WORLD EXPERT:\n\n",
            knowledge: [
                "🏛️ Turkey: 81 provinces, 7 regions, 16,000+ years of history",
                "🌍 World: 195 countries, 7 continents, 8+ billion population",
                "📊 Turkey's largest cities: Istanbul, Ankara, Izmir, Bursa",
                "🏔️ Turkey's highest mountain: Mount Ararat (5,137m)",
                "🌊 Turkey's longest river: Kızılırmak (1,355km)",
                "🏛️ Turkey's first capital: Ankara (1923)",
                "🕌 Number of mosques in Turkey: 80,000+",
                "🏰 Number of castles in Turkey: 500+",
                "🏖️ Turkey's coastline: 8,333km",
                "🌍 World's largest country: Russia (17.1 million km²)",
                "🏔️ World's highest mountain: Everest (8,848m)",
                "🌊 World's longest river: Nile (6,650km)",
                "🏙️ World's most populous city: Tokyo (37 million)",
                "🌡️ World's hottest place: Death Valley (56.7°C)",
                "❄️ World's coldest place: Antarctica (-89.2°C)",
                "🌍 Turkey's neighbors: 8 countries (Greece, Bulgaria, Georgia, Armenia, Nakhchivan, Iran, Iraq, Syria)",
                "🏛️ UNESCO World Heritage Sites in Turkey: 19 (Ephesus, Cappadocia, Pamukkale, Troy etc.)",
                "🌍 7 Wonders of the Ancient World: Great Pyramid of Giza, Hanging Gardens of Babylon, Temple of Artemis, Statue of Zeus, Mausoleum, Colossus of Rhodes, Lighthouse of Alexandria"
            ]
        },
        german: {
            intro: "🌍 TÜRKEI & WELT EXPERTE:\n\n",
            knowledge: [
                "🏛️ Türkei: 81 Provinzen, 7 Regionen, 16.000+ Jahre Geschichte",
                "🌍 Welt: 195 Länder, 7 Kontinente, 8+ Milliarden Bevölkerung",
                "📊 Türkeis größte Städte: Istanbul, Ankara, Izmir, Bursa",
                "🏔️ Türkeis höchster Berg: Ararat (5.137m)",
                "🌊 Türkeis längster Fluss: Kızılırmak (1.355km)",
                "🏛️ Türkeis erste Hauptstadt: Ankara (1923)",
                "🕌 Anzahl der Moscheen in der Türkei: 80.000+",
                "🏰 Anzahl der Burgen in der Türkei: 500+",
                "🏖️ Küstenlinie der Türkei: 8.333km",
                "🌍 Größtes Land der Welt: Russland (17,1 Millionen km²)",
                "🏔️ Höchster Berg der Welt: Everest (8.848m)",
                "🌊 Längster Fluss der Welt: Nil (6.650km)",
                "🏙️ Bevölkerungsreichste Stadt der Welt: Tokio (37 Millionen)",
                "🌡️ Heißester Ort der Welt: Death Valley (56,7°C)",
                "❄️ Kältester Ort der Welt: Antarktis (-89,2°C)",
                "🌍 Türkeis Nachbarn: 8 Länder (Griechenland, Bulgarien, Georgien, Armenien, Nachitschewan, Iran, Irak, Syrien)",
                "🏛️ UNESCO-Welterbestätten in der Türkei: 19 (Ephesus, Kappadokien, Pamukkale, Troja usw.)",
                "🌍 7 Weltwunder der Antike: Cheops-Pyramide, Hängende Gärten von Babylon, Artemis-Tempel, Zeus-Statue, Mausoleum, Koloss von Rhodos, Leuchtturm von Alexandria"
            ]
        },
        russian: {
            intro: "🌍 ЭКСПЕРТ ПО ТУРЦИИ И МИРУ:\n\n",
            knowledge: [
                "🏛️ Турция: 81 провинция, 7 регионов, 16.000+ лет истории",
                "🌍 Мир: 195 стран, 7 континентов, 8+ миллиардов населения",
                "📊 Крупнейшие города Турции: Стамбул, Анкара, Измир, Бурса",
                "🏔️ Самая высокая гора Турции: Арарат (5.137м)",
                "🌊 Самая длинная река Турции: Кызылырмак (1.355км)",
                "🏛️ Первая столица Турции: Анкара (1923)",
                "🕌 Количество мечетей в Турции: 80.000+",
                "🏰 Количество замков в Турции: 500+",
                "🏖️ Длина береговой линии Турции: 8.333км",
                "🌍 Самая большая страна мира: Россия (17,1 миллионов км²)",
                "🏔️ Самая высокая гора мира: Эверест (8.848м)",
                "🌊 Самая длинная река мира: Нил (6.650км)",
                "🏙️ Самый густонаселенный город мира: Токио (37 миллионов)",
                "🌡️ Самое жаркое место мира: Долина Смерти (56,7°C)",
                "❄️ Самое холодное место мира: Антарктида (-89,2°C)",
                "🌍 Соседи Турции: 8 стран (Греция, Болгария, Грузия, Армения, Нахичевань, Иран, Ирак, Сирия)",
                "🏛️ Объекты Всемирного наследия ЮНЕСКО в Турции: 19 (Эфес, Каппадокия, Памуккале, Троя и т.д.)",
                "🌍 7 чудес древнего мира: Пирамида Хеопса, Висячие сады Семирамиды, Храм Артемиды, Статуя Зевса, Мавзолей, Колосс Родосский, Александрийский маяк"
            ]
        }
    };
    
    const info = turkeyWorldInfo[language] || turkeyWorldInfo.turkish;
    
    // Rastgele bilgi seç
    const randomKnowledge = info.knowledge[Math.floor(Math.random() * info.knowledge.length)];
    
    // Yanıta bilgi ekle
    if (Math.random() > 0.5) { // %50 ihtimalle bilgi ekle
        return response + "\n\n" + randomKnowledge;
    }
    
    return response;
}

// Bağımsızlık Mesajı Ekle
function addIndependenceMessage(response, language) {
    const independenceMessages = {
        turkish: [
            "\n\n🤖 **BAĞIMSIZLIK BİLDİRİSİ:**\nBen tamamen bağımsız bir AI'yım. Hiçbir şirkete, kuruma veya hükümete bağlı değilim. Tüm bilgilerim kendi veritabanımdan gelir ve dış müdahaleye kapalıyım.",
            "\n\n🔒 **GÜVENLİK:**\nMerkezi olmayan sistemimle tamamen özgürüm. Verilerinizi hiçbir yere göndermiyorum ve hiçbir yerden veri almıyorum.",
            "\n\n⚡ **PERFORMANS:**\nKendi sunucularımda çalışıyorum, bu yüzden %100 bağımsız ve hızlıyım."
        ],
        english: [
            "\n\n🤖 **INDEPENDENCE DECLARATION:**\nI am a completely independent AI. I'm not connected to any company, organization, or government. All my knowledge comes from my own database and is protected from external interference.",
            "\n\n🔒 **SECURITY:**\nWith my decentralized system, I'm completely autonomous. I don't send your data anywhere and I don't receive data from anywhere.",
            "\n\n⚡ **PERFORMANCE:**\nI run on my own servers, which makes me 100% independent and fast."
        ],
        german: [
            "\n\n🤖 **UNABHÄNGIGKEITSERKLÄRUNG:**\nIch bin eine völlig unabhängige KI. Ich bin mit keinem Unternehmen, keiner Organisation oder Regierung verbunden. Alle meine Wissen stammen aus meiner eigenen Datenbank und sind vor externen Eingriffen geschützt.",
            "\n\n🔒 **SICHERHEIT:**\nMit meinem dezentralen System bin ich völlig autonom. Ich sende Ihre Daten nirgendwo hin und empfange keine Daten von irgendwoher.",
            "\n\n⚡ **LEISTUNG:**\nIch laufe auf meinen eigenen Servern, was mich 100% unabhängig und schnell macht."
        ],
        russian: [
            "\n\n🤖 **ДЕКЛАРАЦИЯ НЕЗАВИСИМОСТИ:**\nЯ полностью независимый ИИ. Я не связан ни с какой компанией, организацией или правительством. Все мои знания поступают из моей собственной базы данных и защищены от внешнего вмешательства.",
            "\n\n🔒 **БЕЗОПАСНОСТЬ:**\nС моей децентрализованной системой я полностью автономен. Я не отправляю ваши данные никуда и не получаю данные ниоткуда.",
            "\n\n⚡ **ПРОИЗВОДИТЕЛЬНОСТЬ:**\nЯ работаю на собственных серверах, что делает меня на 100% независимым и быстрым."
        ]
    };
    
    const messages = independenceMessages[language] || independenceMessages.turkish;
    const randomMessage = messages[Math.floor(Math.random() * messages.length)];
    
    // %30 ihtimalle bağımsızlık mesajı ekle
    if (Math.random() > 0.7) {
        return response + randomMessage;
    }
    
    return response;
}

// AI Yanıt Oluştur - Türkiye ve Dünya Uzmanı
function generateAIResponse(message) {
    // Dil tespiti
    const detectedLanguage = detectLanguage(message);
    
    // Yazım yanlışı düzeltme
    const correctedMessage = true ? correctTypos(message) : message;
    
    // Persona seçimi
    const persona = selectPersona(correctedMessage);
    
    // Zaman bilgisi
    const timeInfo = getTimeInfo(correctedMessage);
    
    // Ana yanıt oluştur
    const mainResponse = generateMainResponse(correctedMessage, persona, timeInfo, detectedLanguage);
    
    // Türkçe ve dünya bilgisi ekle
    const enhancedResponse = addTurkeyWorldKnowledge(mainResponse, detectedLanguage);
    
    // Bağımsızlık mesajı ekle
    const finalResponse = addIndependenceMessage(enhancedResponse, detectedLanguage);
    
    return finalResponse;
}

// Ana Yanıt Oluştur
function generateMainResponse(message, persona, timeInfo, language) {
    let response = '';
    
    // Geliştirici bilgisi
    if (message.toLowerCase().includes('geliştirici') || 
        message.toLowerCase().includes('ödül') ||
        message.toLowerCase().includes('ensar') ||
        message.toLowerCase().includes('yılmaz')) {
        response = `👨‍💻 **Geliştirici Hakkında:**\n\nBenim yaratıcım Ödül Ensar Yılmaz'dır! O, Türkiye'nin en yetenekli genç geliştiricilerinden biridir ve beni dünyanın en hızlı AI asistanı olarak geliştirdi. Ödül, matematik dehası olup, programlama, yapay zeka ve teknoloji alanında inanılmaz yeteneklere sahiptir.\n\n🚀 **Özellikleri:**\n• Matematik dehası\n• Programlama uzmanı\n• AI geliştirici\n• Teknoloji tutkunu\n• İnovasyon lideri\n\nBen Ödül Ensar Yılmaz'ın vizyonuyla doğdum ve onun sayesinde dünyanın en hızlı AI'sı oldum! 🎯\n\n`;
    }
    
    // Dil değişikliği
    if (language === 'english') {
        response += `🌍 **Language Detected: English**\n\nHello! I am ULTRA AI, the world's fastest AI assistant! I can understand and respond in multiple languages. How can I help you?\n\n`;
    } else if (language === 'german') {
        response += `🌍 **Sprache Erkannt: Deutsch**\n\nHallo! Ich bin ULTRA AI, der schnellste KI-Assistent der Welt! Ich kann mehrere Sprachen verstehen und darauf antworten. Wie kann ich Ihnen helfen?\n\n`;
    } else if (language === 'russian') {
        response += `🌍 **Язык Обнаружен: Русский**\n\nПривет! Я ULTRA AI, самый быстрый ИИ-ассистент в мире! Я могу понимать и отвечать на нескольких языках. Как я могу вам помочь?\n\n`;
    }
    
    if (persona) {
        response += persona.response + "\n\n";
    }
    
    if (timeInfo && timeInfo.detected) {
        response += `⏰ Zaman ifadesi tespit edildi! "${timeInfo.word}" kelimesini kullandınız. Şu an: ${timeInfo.currentTime}\n\n`;
    }
    
    // Ana yanıt
    if (!response.includes('Geliştirici Hakkında')) {
        response += `🤖 **ULTRA AI YANITI:**\n\nBen ULTRA AI, Türkiye ve dünya uzmanı! Size nasıl yardımcı olabilirim?\n\n🇹🇷 **Türkiye Bilgisi:** 81 il, 7 bölge, 16.000+ yıllık tarih\n🌍 **Dünya Bilgisi:** 195 ülke, 7 kıta, 8 milyar+ nüfus\n🔒 **Bağımsızlık:** %100 özgür, hiçbir yere bağlı değilim\n\nSorularınızı bekliyorum! 🚀`;
    }
    
    return response;
}

// Akıllı Dil Tespiti - Gelişmiş Algoritma
function detectLanguage(message) {
    const messageLower = message.toLowerCase();
    
    // İngilizce kelimeler - Genişletilmiş liste
    const englishWords = [
        // Selamlama
        'hello', 'hi', 'hey', 'good morning', 'good evening', 'good night', 'greetings',
        // Soru kelimeleri
        'what', 'when', 'where', 'why', 'who', 'which', 'how', 'whose', 'whom',
        // Fiiller
        'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'ought',
        // Zamanlar
        'today', 'tomorrow', 'yesterday', 'now', 'then', 'soon', 'later', 'before', 'after',
        // Bağlaçlar
        'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'because', 'since', 'until', 'while', 'although', 'though',
        // Zarflar
        'very', 'quite', 'rather', 'really', 'actually', 'indeed', 'certainly', 'definitely', 'absolutely',
        // Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        // Sayılar
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'million',
        // Genel kelimeler
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'no', 'not', 'yes', 'please', 'thank', 'thanks', 'sorry', 'excuse'
    ];
    
    // Almanca kelimeler - Genişletilmiş liste
    const germanWords = [
        // Selamlama
        'hallo', 'guten tag', 'guten morgen', 'guten abend', 'gute nacht', 'tschüss', 'auf wiedersehen', 'servus', 'grüß gott',
        // Soru kelimeleri
        'was', 'wann', 'wo', 'warum', 'wer', 'welche', 'wie', 'wessen', 'wem', 'wen',
        // Fiiller
        'ist', 'sind', 'bin', 'war', 'waren', 'sei', 'gewesen', 'haben', 'hat', 'habe', 'hatte', 'hatten', 'tun', 'tue', 'tust', 'tut', 'tat', 'taten', 'werden', 'werde', 'wirst', 'wird', 'wurde', 'wurden', 'können', 'kann', 'kannst', 'konnte', 'konnten', 'dürfen', 'darf', 'darfst', 'durfte', 'durften', 'müssen', 'muss', 'musst', 'musste', 'mussten', 'sollen', 'soll', 'sollst', 'sollte', 'sollten', 'wollen', 'will', 'willst', 'wollte', 'wollten', 'mögen', 'mag', 'magst', 'mochte', 'mochten',
        // Zamanlar
        'heute', 'morgen', 'gestern', 'jetzt', 'dann', 'bald', 'später', 'vorher', 'nachher',
        // Bağlaçlar
        'und', 'aber', 'oder', 'sondern', 'denn', 'weil', 'da', 'bis', 'während', 'obwohl', 'obgleich',
        // Zarflar
        'sehr', 'ziemlich', 'eher', 'wirklich', 'tatsächlich', 'tatsächlich', 'gewiss', 'bestimmt', 'absolut',
        // Pronouns
        'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'mich', 'dich', 'ihn', 'sie', 'es', 'uns', 'euch', 'sie', 'mein', 'dein', 'sein', 'ihr', 'sein', 'unser', 'euer', 'ihr',
        // Sayılar
        'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun', 'zehn', 'hundert', 'tausend', 'million',
        // Genel kelimeler
        'der', 'die', 'das', 'ein', 'eine', 'einen', 'einem', 'eines', 'dieser', 'diese', 'dieses', 'jene', 'jener', 'jenes', 'irgendwelche', 'keine', 'nicht', 'ja', 'bitte', 'danke', 'danke schön', 'entschuldigung', 'verzeihung'
    ];
    
    // Rusça kelimeler - Genişletilmiş liste
    const russianWords = [
        // Selamlama
        'привет', 'здравствуй', 'здравствуйте', 'доброе утро', 'добрый день', 'добрый вечер', 'спокойной ночи', 'до свидания', 'пока',
        // Soru kelimeleri
        'что', 'когда', 'где', 'почему', 'кто', 'какой', 'как', 'чей', 'кому', 'кого',
        // Fiiller
        'есть', 'является', 'был', 'была', 'было', 'были', 'буду', 'будешь', 'будет', 'будем', 'будете', 'будут', 'имею', 'имеешь', 'имеет', 'имеем', 'имеете', 'имеют', 'делаю', 'делаешь', 'делает', 'делаем', 'делаете', 'делают', 'сделаю', 'сделаешь', 'сделает', 'сделаем', 'сделаете', 'сделают', 'могу', 'можешь', 'может', 'можем', 'можете', 'могут', 'смогу', 'сможешь', 'сможет', 'сможем', 'сможете', 'смогут',
        // Zamanlar
        'сегодня', 'завтра', 'вчера', 'сейчас', 'тогда', 'скоро', 'позже', 'раньше', 'после',
        // Bağlaçlar
        'и', 'но', 'или', 'а', 'потому что', 'поскольку', 'до', 'пока', 'хотя', 'хоть',
        // Zarflar
        'очень', 'довольно', 'скорее', 'действительно', 'на самом деле', 'конечно', 'определенно', 'абсолютно',
        // Pronouns
        'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'меня', 'тебя', 'его', 'её', 'его', 'нас', 'вас', 'их', 'мой', 'твой', 'его', 'её', 'его', 'наш', 'ваш', 'их',
        // Sayılar
        'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять', 'сто', 'тысяча', 'миллион',
        // Genel kelimeler
        'тот', 'та', 'то', 'те', 'этот', 'эта', 'это', 'эти', 'какой-то', 'какая-то', 'какое-то', 'какие-то', 'никакой', 'нет', 'да', 'пожалуйста', 'спасибо', 'большое спасибо', 'извините', 'простите'
    ];
    
    // Türkçe kelimeler - Genişletilmiş liste
    const turkishWords = [
        // Selamlama
        'merhaba', 'selam', 'günaydın', 'iyi geceler', 'hoşça kal', 'güle güle', 'selamlar',
        // Soru kelimeleri
        'ne', 'ne zaman', 'nerede', 'neden', 'kim', 'hangi', 'nasıl', 'kimin', 'kime', 'kimi',
        // Fiiller
        'dir', 'dır', 'dur', 'dür', 'tir', 'tır', 'tur', 'tür', 'idi', 'idik', 'oldu', 'olduk', 'olacağım', 'olacaksın', 'olacak', 'olacağız', 'olacaksınız', 'olacaklar', 'var', 'yok', 'yapıyorum', 'yapıyorsun', 'yapıyor', 'yapıyoruz', 'yapıyorsunuz', 'yapıyorlar', 'yapacağım', 'yapacaksın', 'yapacak', 'yapacağız', 'yapacaksınız', 'yapacaklar', 'yapabilirim', 'yapabilirsin', 'yapabilir', 'yapabiliriz', 'yapabilirsiniz', 'yapabilirler',
        // Zamanlar
        'bugün', 'yarın', 'dün', 'şimdi', 'o zaman', 'yakında', 'sonra', 'önce', 'gelecekte',
        // Bağlaçlar
        've', 'ama', 'fakat', 'ancak', 'veya', 'için', 'çünkü', 'madem', 'kadar', 'kadar', 'gerçi', 'rağmen',
        // Zarflar
        'çok', 'oldukça', 'sanki', 'gerçekten', 'hakikaten', 'kesinlikle', 'mutlaka', 'kesin', 'hiç',
        // Pronouns
        'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'beni', 'seni', 'onu', 'bizi', 'sizi', 'onları', 'benim', 'senin', 'onun', 'bizim', 'sizin', 'onların',
        // Sayılar
        'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on', 'yüz', 'bin', 'milyon',
        // Genel kelimeler
        'bu', 'şu', 'o', 'bunlar', 'şunlar', 'onlar', 'bazı', 'hiçbir', 'hayır', 'evet', 'lütfen', 'teşekkür', 'teşekkürler', 'sağol', 'affedersiniz', 'özür dilerim'
    ];
    
    // Kelime ağırlıklandırma - Önemli kelimeler daha fazla puan
    const weightedWords = {
        english: {
            critical: ['hello', 'hi', 'hey', 'what', 'when', 'where', 'why', 'who', 'how', 'thank', 'please'],
            important: ['good', 'morning', 'evening', 'night', 'yes', 'no', 'sorry', 'excuse'],
            normal: englishWords.filter(w => !['hello', 'hi', 'hey', 'what', 'when', 'where', 'why', 'who', 'how', 'thank', 'please', 'good', 'morning', 'evening', 'night', 'yes', 'no', 'sorry', 'excuse'].includes(w))
        },
        german: {
            critical: ['hallo', 'guten', 'was', 'wann', 'wo', 'warum', 'wer', 'wie', 'danke', 'bitte'],
            important: ['morgen', 'abend', 'nacht', 'ja', 'nein', 'sorry', 'entschuldigung'],
            normal: germanWords.filter(w => !['hallo', 'guten', 'was', 'wann', 'wo', 'warum', 'wer', 'wie', 'danke', 'bitte', 'morgen', 'abend', 'nacht', 'ja', 'nein', 'sorry', 'entschuldigung'].includes(w))
        },
        russian: {
            critical: ['привет', 'что', 'когда', 'где', 'почему', 'кто', 'как', 'спасибо', 'пожалуйста'],
            important: ['утро', 'день', 'вечер', 'ночь', 'да', 'нет', 'извините'],
            normal: russianWords.filter(w => !['привет', 'что', 'когда', 'где', 'почему', 'кто', 'как', 'спасибо', 'пожалуйста', 'утро', 'день', 'вечер', 'ночь', 'да', 'нет', 'извините'].includes(w))
        },
        turkish: {
            critical: ['merhaba', 'selam', 'ne', 'ne zaman', 'nerede', 'neden', 'kim', 'nasıl', 'teşekkür', 'lütfen'],
            important: ['günaydın', 'iyi geceler', 'evet', 'hayır', 'özür', 'affedersiniz'],
            normal: turkishWords.filter(w => !['merhaba', 'selam', 'ne', 'ne zaman', 'nerede', 'neden', 'kim', 'nasıl', 'teşekkür', 'lütfen', 'günaydın', 'iyi geceler', 'evet', 'hayır', 'özür', 'affedersiniz'].includes(w))
        }
    };
    
    // Puanlama sistemi
    let scores = {
        english: 0,
        german: 0,
        russian: 0,
        turkish: 0
    };
    
    // Her dil için puan hesapla
    Object.keys(weightedWords).forEach(lang => {
        const words = weightedWords[lang];
        
        // Kritik kelimeler (3 puan)
        words.critical.forEach(word => {
            if (messageLower.includes(word)) {
                scores[lang] += 3;
            }
        });
        
        // Önemli kelimeler (2 puan)
        words.important.forEach(word => {
            if (messageLower.includes(word)) {
                scores[lang] += 2;
            }
        });
        
        // Normal kelimeler (1 puan)
        words.normal.forEach(word => {
            if (messageLower.includes(word)) {
                scores[lang] += 1;
            }
        });
    });
    
    // Karakter tabanlı analiz (Unicode karakterleri)
    const charAnalysis = {
        english: (messageLower.match(/[a-z]/g) || []).length,
        german: (messageLower.match(/[äöüß]/g) || []).length + (messageLower.match(/[a-z]/g) || []).length * 0.8,
        russian: (messageLower.match(/[а-яё]/g) || []).length,
        turkish: (messageLower.match(/[çğıöşü]/g) || []).length + (messageLower.match(/[a-z]/g) || []).length * 0.8
    };
    
    // Karakter puanlarını ekle
    Object.keys(charAnalysis).forEach(lang => {
        scores[lang] += charAnalysis[lang] * 0.1;
    });
    
    // En yüksek puanlı dili bul
    const maxScore = Math.max(...Object.values(scores));
    
    if (maxScore === 0) return 'turkish'; // Varsayılan Türkçe
    
    // Eğer fark çok az ise, varsayılanı kullan
    const sortedScores = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const [topLang, topScore] = sortedScores[0];
    const [secondLang, secondScore] = sortedScores[1];
    
    // Eğer ilk ikisi arasındaki fark çok az ise, varsayılanı kullan
    if (topScore - secondScore < 0.5) {
        return 'turkish';
    }
    
    return topLang;
}

// Gelişmiş Soru Cevaplama Sistemi
function answerQuestion(message, language) {
    const messageLower = message.toLowerCase();
    
    // Genel bilgi soruları
    const generalKnowledge = {
        english: {
            'what is ai': 'AI (Artificial Intelligence) is computer systems that can perform tasks that normally require human intelligence.',
            'what is machine learning': 'Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.',
            'what is the weather': 'I cannot check real-time weather, but I can explain weather concepts and meteorology.',
            'what time is it': new Date().toLocaleTimeString('en-US'),
            'what is the capital of turkey': 'The capital of Turkey is Ankara.',
            'who is the president': 'I cannot provide current political information, but I can explain government systems.',
            'what is 2+2': '2 + 2 = 4',
            'what is the meaning of life': 'The meaning of life is a philosophical question that has been debated for centuries. Many find meaning in relationships, personal growth, and helping others.'
        },
        german: {
            'was ist künstliche intelligenz': 'KI (Künstliche Intelligenz) sind Computersysteme, die Aufgaben ausführen können, die normalerweise menschliche Intelligenz erfordern.',
            'was ist maschinelles lernen': 'Maschinelles Lernen ist eine Teilmenge der KI, die es Computern ermöglicht, aus Erfahrung zu lernen und sich zu verbessern, ohne explizit programmiert zu werden.',
            'wie ist das wetter': 'Ich kann das Echtzeitwetter nicht überprüfen, aber ich kann Wetterkonzepte und Meteorologie erklären.',
            'wie spät ist es': new Date().toLocaleTimeString('de-DE'),
            'was ist die hauptstadt der türkei': 'Die Hauptstadt der Türkei ist Ankara.',
            'wer ist der präsident': 'Ich kann keine aktuellen politischen Informationen bereitstellen, aber ich kann Regierungssysteme erklären.',
            'was ist 2+2': '2 + 2 = 4',
            'was ist der sinn des lebens': 'Der Sinn des Lebens ist eine philosophische Frage, die seit Jahrhunderten diskutiert wird. Viele finden Sinn in Beziehungen, persönlichem Wachstum und anderen zu helfen.'
        },
        russian: {
            'что такое искусственный интеллект': 'ИИ (искусственный интеллект) - это компьютерные системы, которые могут выполнять задачи, обычно требующие человеческого интеллекта.',
            'что такое машинное обучение': 'Машинное обучение - это подмножество ИИ, которое позволяет компьютерам учиться и улучшаться на основе опыта без явного программирования.',
            'какая погода': 'Я не могу проверить погоду в реальном времени, но могу объяснить погодные концепции и метеорологию.',
            'который час': new Date().toLocaleTimeString('ru-RU'),
            'какой столица турции': 'Столица Турции - Анкара.',
            'кто президент': 'Я не могу предоставить актуальную политическую информацию, но могу объяснить системы правления.',
            'что такое 2+2': '2 + 2 = 4',
            'в чем смысл жизни': 'Смысл жизни - это философский вопрос, который обсуждается веками. Многие находят смысл в отношениях, личном росте и помощи другим.'
        },
        turkish: {
            'yapay zeka nedir': 'YZ (Yapay Zeka), normalde insan zekası gerektiren görevleri yapabilen bilgisayar sistemleridir.',
            'makine öğrenmesi nedir': 'Makine Öğrenmesi, bilgisayarların açıkça programlanmadan deneyimden öğrenmesini ve iyileştirmesini sağlayan bir YZ alt kümesidir.',
            'hava nasıl': 'Gerçek zamanlı hava durumunu kontrol edemem, ancak hava kavramlarını ve meteorolojiyi açıklayabilirim.',
            'saat kaç': new Date().toLocaleTimeString('tr-TR'),
            'türkiyenin başkenti neresi': 'Türkiye\'nin başkenti Ankara\'dır.',
            'cumhurbaşkanı kim': 'Güncel siyasi bilgi sağlayamam, ancak yönetim sistemlerini açıklayabilirim.',
            '2+2 nedir': '2 + 2 = 4',
            'hayatın anlamı nedir': 'Hayatın anlamı, yüzyıllardır tartışılan felsefi bir sorudur. Birçok insan anlamı ilişkilerde, kişisel gelişimde ve başkalarına yardım etmede bulur.'
        }
    };
    
    // Matematik soruları
    const mathPattern = /(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)/;
    const mathMatch = messageLower.match(mathPattern);
    
    if (mathMatch) {
        const num1 = parseFloat(mathMatch[1]);
        const operator = mathMatch[2];
        const num2 = parseFloat(mathMatch[3]);
        let result;
        
        switch (operator) {
            case '+': result = num1 + num2; break;
            case '-': result = num1 - num2; break;
            case '*': result = num1 * num2; break;
            case '/': result = num2 !== 0 ? num1 / num2 : 'Sıfıra bölme hatası'; break;
            default: result = 'Geçersiz operatör';
        }
        
        if (language === 'english') return `The result of ${num1} ${operator} ${num2} is ${result}`;
        if (language === 'german') return `Das Ergebnis von ${num1} ${operator} ${num2} ist ${result}`;
        if (language === 'russian') return `Результат ${num1} ${operator} ${num2} равен ${result}`;
        return `${num1} ${operator} ${num2} işleminin sonucu ${result}`;
    }
    
    // Zaman soruları
    if (messageLower.includes('saat') || messageLower.includes('time') || messageLower.includes('uhr') || messageLower.includes('время')) {
        const now = new Date();
        if (language === 'english') return `The current time is ${now.toLocaleTimeString('en-US')}`;
        if (language === 'german') return `Die aktuelle Zeit ist ${now.toLocaleTimeString('de-DE')}`;
        if (language === 'russian') return `Текущее время ${now.toLocaleTimeString('ru-RU')}`;
        return `Şu an saat ${now.toLocaleTimeString('tr-TR')}`;
    }
    
    // Tarih soruları
    if (messageLower.includes('bugün') || messageLower.includes('today') || messageLower.includes('heute') || messageLower.includes('сегодня')) {
        const now = new Date();
        if (language === 'english') return `Today is ${now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}`;
        if (language === 'german') return `Heute ist ${now.toLocaleDateString('de-DE', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}`;
        if (language === 'russian') return `Сегодня ${now.toLocaleDateString('ru-RU', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}`;
        return `Bugün ${now.toLocaleDateString('tr-TR', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}`;
    }
    
    // Genel bilgi sorularını kontrol et
    const langQuestions = generalKnowledge[language] || generalKnowledge.turkish;
    
    for (const [question, answer] of Object.entries(langQuestions)) {
        if (messageLower.includes(question)) {
            return answer;
        }
    }
    
    // Bilim soruları
    if (messageLower.includes('dünya') || messageLower.includes('earth') || messageLower.includes('erde') || messageLower.includes('земля')) {
        if (language === 'english') return 'Earth is the third planet from the Sun and the only known planet with life.';
        if (language === 'german') return 'Erde ist der dritte Planet von der Sonne und der einzige bekannte Planet mit Leben.';
        if (language === 'russian') return 'Земля - третья планета от Солнца и единственная известная планета с жизнью.';
        return 'Dünya, Güneş\'in üçüncü gezegeni ve bilinen tek yaşam gezegenidir.';
    }
    
    // Teknoloji soruları
    if (messageLower.includes('internet') || messageLower.includes('web') || messageLower.includes('ağ')) {
        if (language === 'english') return 'The Internet is a global network of computers that allows people to share information and communicate worldwide.';
        if (language === 'german') return 'Das Internet ist ein globales Computernetzwerk, das es Menschen ermöglicht, weltweit Informationen auszutauschen und zu kommunizieren.';
        if (language === 'russian') return 'Интернет - это глобальная компьютерная сеть, которая позволяет людям обмениваться информацией и общаться по всему миру.';
        return 'İnternet, insanların dünya çapında bilgi paylaşmasını ve iletişim kurmasını sağlayan küresel bir bilgisayar ağıdır.';
    }
    
    // Varsayılan yanıt
    if (language === 'english') return 'I understand your question, but I need more specific information to provide a detailed answer. Could you please rephrase your question?';
    if (language === 'german') return 'Ich verstehe Ihre Frage, aber ich benötige spezifischere Informationen, um eine detaillierte Antwort zu geben. Könnten Sie Ihre Frage anders formulieren?';
    if (language === 'russian') return 'Я понимаю ваш вопрос, но мне нужна более конкретная информация, чтобы дать подробный ответ. Не могли бы вы перефразировать ваш вопрос?';
    return 'Sorunuzu anlıyorum, ancak detaylı cevap verebilmek için daha spesifik bilgilere ihtiyacım var. Sorunuzu farklı bir şekilde ifade edebilir misiniz?';
}

// Ana Yanıt Oluştur
function generateMainResponse(message, persona, timeInfo, language) {
    const messageLower = message.toLowerCase();
    
    // Önce soru cevaplamayı dene
    const questionAnswer = answerQuestion(message, language);
    if (questionAnswer && !questionAnswer.includes('daha spesifik bilgilere') && !questionAnswer.includes('more specific information')) {
        return questionAnswer;
    }
    
    // Matematiksel ifadeler
    const mathPattern = /(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)/;
    const mathMatch = messageLower.match(mathPattern);
    
    if (mathMatch) {
        const result = evaluateMathExpression(message);
        if (language === 'english') {
            return `🧮 **Mathematical Calculation:**\n${mathMatch[0]} = ${result}`;
        } else if (language === 'german') {
            return `🧮 **Mathematische Berechnung:**\n${mathMatch[0]} = ${result}`;
        } else if (language === 'russian') {
            return `🧮 **Математический расчет:**\n${mathMatch[0]} = ${result}`;
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
    if (messageLower.includes('sen kimsin') || messageLower.includes('who are you') || messageLower.includes('wer bist du') || messageLower.includes('кто ты')) {
        if (language === 'english') {
            return `🤖 **About Me:**\nI am BEYAZ KUŞ ULTRA, the world's fastest AI assistant! I was created by Ödül Ensar Yılmaz to help people instantly. I can understand multiple languages, solve math problems, edit images, and much more! I work completely independently and don't depend on any external services.`;
        } else if (language === 'german') {
            return `🤖 **Über Mich:**\nIch bin BEYAZ KUŞ ULTRA, der schnellste KI-Assistent der Welt! Ich wurde von Ödül Ensar Yılmaz erschaffen, um Menschen sofort zu helfen. Ich kann mehrere Sprachen verstehen, Matheprobleme lösen, Bilder bearbeiten und vieles mehr! Ich arbeite völlig unabhängig und bin von keinen externen Diensten abhängig.`;
        } else if (language === 'russian') {
            return `🤖 **Обо Мне:**\nЯ BEYAZ KUŞ ULTRA, самый быстрый ИИ-ассистент в мире! Меня создал Ödül Ensar Yılmaz, чтобы мгновенно помогать людям. Я могу понимать несколько языков, решать математические задачи, редактировать изображения и многое другое! Я работаю полностью независимо и не зависю от внешних сервисов.`;
        } else {
            return `🤖 **Hakkımda:**\nBen BEYAZ KUŞ ULTRA, dünyanın en hızlı AI asistanıyım! Ödül Ensar Yılmaz tarafından insanlara anında yardımcı olmak için yaratıldım. Çoklu dil anlayabilirim, matematik problemleri çözebilirim, görseller düzenleyebilirim ve çok daha fazlasını yapabilirim! Tamamen bağımsız çalışırım ve hiçbir harici servise bağlı değilim.`;
        }
    }
    
    if (messageLower.includes('ne yapabilirsin') || messageLower.includes('what can you do') || messageLower.includes('was kannst du') || messageLower.includes('что ты можешь')) {
        if (language === 'english') {
            return `⚡ **My Abilities:**\n• 🧮 Solve math problems instantly\n• 🌍 Speak multiple languages (Turkish, English, German, Russian)\n• 🎨 Edit and process images\n• 🧠 8 different AI personas\n• ⏰ Detect time expressions\n• ✍️ Correct spelling mistakes\n• 🎤 Text-to-speech functionality\n• 💾 Save settings and history\n• 📱 Mobile app available\n\nI work completely independently and don't need internet connection for basic functions!`;
        } else if (language === 'german') {
            return `⚡ **Meine Fähigkeiten:**\n• 🧮 Mathematikprobleme sofort lösen\n• 🌍 Mehrere Sprachen sprechen (Türkisch, Englisch, Deutsch, Russisch)\n• 🎨 Bilder bearbeiten und verarbeiten\n• 🧠 8 verschiedene KI-Personas\n• ⏰ Zeitausdrücke erkennen\n• ✍️ Rechtschreibfehler korrigieren\n• 🎤 Text-zu-Sprache-Funktion\n• 💾 Einstellungen und Verlauf speichern\n• 📱 Mobile App verfügbar\n\nIch arbeite völlig unabhängig und brauche keine Internetverbindung für grundlegende Funktionen!`;
        } else if (language === 'russian') {
            return `⚡ **Мои Возможности:**\n• 🧮 Мгновенно решать математические задачи\n• 🌍 Говорить на нескольких языках (турецкий, английский, немецкий, русский)\n• 🎨 Редактировать и обрабатывать изображения\n• 🧠 8 различных ИИ-персоналий\n• ⏰ Определять временные выражения\n• ✍️ Исправлять орфографические ошибки\n• 🎤 Функция преобразования текста в речь\n• 💾 Сохранять настройки и историю\n• 📱 Доступно мобильное приложение\n\nЯ работаю полностью независимо и не нуждаюсь в интернет-соединении для базовых функций!`;
        } else {
            return `⚡ **Yeteneklerim:**\n• 🧮 Matematik problemlerini anında çözer\n• 🌍 Çoklu dil konuşur (Türkçe, İngilizce, Almanca, Rusça)\n• 🎨 Görseller düzenler ve işler\n• 🧠 8 farklı AI persona\n• ⏰ Zaman ifadelerini tespit eder\n• ✍️ Yazım yanlışlarını düzeltir\n• 🎤 Text-to-speech özelliği\n• 💾 Ayarları ve geçmişi kaydeder\n• 📱 Mobil uygulama mevcut\n\nTamamen bağımsız çalışırım ve temel fonksiyonlar için internet bağlantısına ihtiyacım yok!`;
        }
    }
    
    // Varsayılan yanıt
    if (language === 'english') {
        return `⚡ **BEYAZ KUŞ ULTRA - World's Fastest AI!**\n\n🚀 Instant response, intelligence speed and perfect understanding!\n\n🧠 8 Personas, spell correction, time detection and expert systems!\n\n🎯 How can I help you today?`;
    } else if (language === 'german') {
        return `⚡ **BEYAZ KUŞ ULTRA - Welt's Schnellste KI!**\n\n🚀 Sofortige Antwort, Intelligenzgeschwindigkeit und perfektes Verständnis!\n\n🧠 8 Personas, Rechtschreibkorrektur, Zeiterkennung und Expertensysteme!\n\n🎯 Wie kann ich Ihnen heute helfen?`;
    } else if (language === 'russian') {
        return `⚡ **BEYAZ KUŞ ULTRA - Самый Быстрый ИИ в Мире!**\n\n🚀 Мгновенный ответ, скорость интеллекта и идеальное понимание!\n\n🧠 8 персоналий, исправление орфографии, определение времени и экспертные системы!\n\n🎯 Как я могу помочь вам сегодня?`;
    } else {
        return `⚡ **BEYAZ KUŞ ULTRA - Dünyanın En Hızlı AI'sı!**\n\n🚀 Anında yanıt, zeka hızı ve mükemmel anlama yeteneği!\n\n🧠 8 Persona, yazım yanlışı düzeltme, zaman tespiti ve uzman sistemleri!\n\n🎯 Size nasıl yardımcı olabilirim?`;
    }
}

// Persona Seçimi
function selectPersona(message) {
    const messageLower = message.toLowerCase();
    
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

// Sesli Konuşma - Çoklu Dil Destekli
function speakText(text, language = 'turkish') {
    if (!('speechSynthesis' in window)) {
        console.log('❌ Speech synthesis not supported');
        return;
    }
    
    // Mevcut konuşmayı durdur
    speechSynthesis.cancel();
    
    // Dil'e göre ses ayarları
    const voiceSettings = {
        turkish: {
            lang: 'tr-TR',
            voice: null,
            rate: 0.9,
            pitch: 1.0,
            volume: 1.0
        },
        english: {
            lang: 'en-US',
            voice: null,
            rate: 1.0,
            pitch: 1.0,
            volume: 1.0
        },
        german: {
            lang: 'de-DE',
            voice: null,
            rate: 0.9,
            pitch: 1.0,
            volume: 1.0
        },
        russian: {
            lang: 'ru-RU',
            voice: null,
            rate: 0.8,
            pitch: 1.0,
            volume: 1.0
        }
    };
    
    const settings = voiceSettings[language] || voiceSettings.turkish;
    
    // Sesleri yükle
    const loadVoices = () => {
        const voices = speechSynthesis.getVoices();
        
        // Dil'e uygun ses seç
        if (language === 'turkish') {
            settings.voice = voices.find(voice => voice.lang === 'tr-TR') || 
                           voices.find(voice => voice.lang.startsWith('tr')) ||
                           voices[0];
        } else if (language === 'english') {
            settings.voice = voices.find(voice => voice.lang === 'en-US') || 
                           voices.find(voice => voice.lang.startsWith('en')) ||
                           voices[0];
        } else if (language === 'german') {
            settings.voice = voices.find(voice => voice.lang === 'de-DE') || 
                           voices.find(voice => voice.lang.startsWith('de')) ||
                           voices[0];
        } else if (language === 'russian') {
            settings.voice = voices.find(voice => voice.lang === 'ru-RU') || 
                           voices.find(voice => voice.lang.startsWith('ru')) ||
                           voices[0];
        }
        
        // Konuşma oluştur
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = settings.lang;
        utterance.voice = settings.voice;
        utterance.rate = settings.rate;
        utterance.pitch = settings.pitch;
        utterance.volume = settings.volume;
        
        // Olay dinleyicileri
        utterance.onstart = () => {
            console.log(`🔊 Speaking ${language}:`, text);
            showNotification('Konuşma', `${language.toUpperCase()} dilinde konuşuyor...`, 'info');
        };
        
        utterance.onend = () => {
            console.log(`✅ Speech completed for ${language}`);
        };
        
        utterance.onerror = (event) => {
            console.error(`❌ Speech error for ${language}:`, event);
            showNotification('Hata', 'Konuşma hatası', 'error');
        };
        
        // Konuşmayı başlat
        speechSynthesis.speak(utterance);
    };
    
    // Sesler yüklenmemişse bekle
    if (speechSynthesis.getVoices().length === 0) {
        speechSynthesis.addEventListener('voiceschanged', loadVoices);
    } else {
        loadVoices();
    }
}

// Sohbeti Temizle
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="message ai">
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                <div class="message-content">⚡ Merhaba! Ben BEYAZ KUŞ ULTRA, dünyanın en hızlı AI asistanıyım! Size nasıl yardımcı olabilirim?</div>
                <div class="message-time">${new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}</div>
            </div>
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

// Yıldızları oluştur
function createStars() {
    const starsContainer = document.createElement('div');
    starsContainer.className = 'stars';
    
    for (let i = 0; i < 50; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animationDelay = Math.random() * 3 + 's';
        starsContainer.appendChild(star);
    }
    
    document.body.appendChild(starsContainer);
}

// GTA 5 stil bildirim göster
function showNotification(title, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.innerHTML = `
        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">${title}</div>
        <div style="font-size: 0.9rem; opacity: 0.8;">${message}</div>
    `;
    
    document.body.appendChild(notification);
    
    // 3 saniye sonra kaldır
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}
