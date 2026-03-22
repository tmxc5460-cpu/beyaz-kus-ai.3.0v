// BEYAZ KUŞ Service Worker - PWA ve Bildirimler
const CACHE_NAME = 'beyaz-kus-v1.0';
const urlsToCache = [
    '/',
    '/index.html',
    '/app.html',
    '/script.js',
    '/manifest.json',
    'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
];

// Service Worker Kurulumu
self.addEventListener('install', function(event) {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(function(cache) {
                return cache.addAll(urlsToCache);
            })
    );
});

// Service Worker Aktifleşme
self.addEventListener('activate', function(event) {
    event.waitUntil(
        caches.keys().then(function(cacheNames) {
            return Promise.all(
                cacheNames.map(function(cacheName) {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// İstekleri Yakala ve Cache'ten Yanıt Ver
self.addEventListener('fetch', function(event) {
    event.respondWith(
        caches.match(event.request)
            .then(function(response) {
                // Cache'te varsa oradan yanıt ver
                if (response) {
                    return response;
                }
                
                // Cache'te yoksa ağdan iste
                return fetch(event.request).then(
                    function(response) {
                        // Geçerli yanıt değilse direkt geri dön
                        if(!response || response.status !== 200 || response.type !== 'basic') {
                            return response;
                        }
                        
                        // Yanıtı cache'e ekle
                        var responseToCache = response.clone();
                        caches.open(CACHE_NAME)
                            .then(function(cache) {
                                cache.put(event.request, responseToCache);
                            });
                        
                        return response;
                    }
                );
            })
    );
});

// Push Bildirimleri
self.addEventListener('push', function(event) {
    const options = {
        body: event.data ? event.data.text() : 'BEYAZ KUŞ size bir mesaj gönderdi!',
        icon: '/icon-192.png',
        badge: '/badge-72.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'Uygulamayı Aç',
                icon: '/checkmark.png'
            },
            {
                action: 'close',
                title: 'Kapat',
                icon: '/xmark.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('BEYAZ KUŞ', options)
    );
});

// Bildirim Tıklama
self.addEventListener('notificationclick', function(event) {
    event.notification.close();
    
    if (event.action === 'explore') {
        // Uygulamayı aç
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});
