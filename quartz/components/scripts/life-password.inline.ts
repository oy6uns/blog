// Life 폴더 암호 보호
(function() {
    let passwordProtectionActive = false;
    let checkInterval;
    
    function checkAndProtectLifePage() {
        const currentPath = window.location.pathname;
        console.log('🔍 Current path:', currentPath);
        
        // ===== URL 패턴 설정 =====
        const isLifePage = /\/[Ll]ife($|\/)/i.test(currentPath);
        
        // 다른 패턴 예시들:
        // 1. 여러 폴더 보호: const isLifePage = /\/(life|private|secret)($|\/)/i.test(currentPath);
        // 2. 특정 파일만: const isLifePage = /\/secret-diary\.html$/.test(currentPath);
        // 3. 하위 폴더 포함: const isLifePage = /\/life\//i.test(currentPath);
        // 4. 정확한 경로만: const isLifePage = currentPath === '/life' || currentPath === '/life/';
        // ============================================
        
        console.log('🔍 Is Life page:', isLifePage);
        
        if (!isLifePage) {
            console.log('❌ Not a Life page, skipping password protection');
            // Life 페이지가 아니면 기존 보호 제거
            removePasswordProtection();
            return false;
        }
        
        console.log('✅ Life page detected!');
        
        // 이미 인증된 경우 스킵
        const hasAccess = sessionStorage.getItem('life-auth') === 'granted';
        console.log('🔍 Has access:', hasAccess);
        
        if (hasAccess) {
            console.log('✅ Already authenticated, skipping');
            removePasswordProtection();
            return false;
        }
        
        if (passwordProtectionActive) {
            console.log('🔒 Password protection already active');
            return true;
        }
        
        console.log('🔒 Need authentication, showing password prompt');
        showPasswordPrompt();
        return true;
    }
    
    function removePasswordProtection() {
        const existingProtection = document.getElementById('password-protection');
        if (existingProtection) {
            existingProtection.remove();
        }
        passwordProtectionActive = false;
        
        // 콘텐츠 블러 제거
        const elements = document.querySelectorAll('body > *:not(#password-protection)');
        elements.forEach(el => {
            const element = el as HTMLElement;
            if (element.style) {
                element.style.filter = '';
                element.style.pointerEvents = '';
                element.style.userSelect = '';
            }
        });
    }
    
    function showPasswordPrompt() {
        console.log('showPasswordPrompt called');
        
        // 이미 암호 창이 있으면 리턴
        if (document.getElementById('password-protection')) {
            console.log('❌ Password protection already exists');
            return;
        }
        
        passwordProtectionActive = true;
        console.log('✅ Setting passwordProtectionActive to true');
        
        // 페이지 콘텐츠를 블러 처리
        const bodyChildren = document.querySelectorAll('body > *');
        console.log('🔍 Found', bodyChildren.length, 'body children');
        
        bodyChildren.forEach(el => {
            const element = el as HTMLElement;
            if (element.tagName !== 'SCRIPT' && element.tagName !== 'STYLE' && element.style) {
                element.style.filter = 'blur(5px)';
                element.style.pointerEvents = 'none';
                element.style.userSelect = 'none';
                console.log('🌫️ Blurring element:', element.tagName, element.className);
            }
        });
        
        // 암호 입력 폼 생성
        const passwordDiv = document.createElement('div');
        passwordDiv.id = 'password-protection';
        passwordDiv.innerHTML = `
            <div style="
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
                max-width: 350px;
                width: 90%;
                z-index: 9999;
                border: 2px solid #007bff;
                animation: slideDown 0.3s ease-out;
            ">
                <style>
                    @keyframes slideDown {
                        from {
                            opacity: 0;
                            transform: translateX(-50%) translateY(-20px);
                        }
                        to {
                            opacity: 1;
                            transform: translateX(-50%) translateY(0);
                        }
                    }
                </style>
                <div style="text-align: center; margin-bottom: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔐</div>
                    <h3 style="margin: 0; color: #333; font-size: 1.1rem;">Life 폴더 접근</h3>
                    <p style="margin: 0.5rem 0 0; color: #666; font-size: 0.9rem;">
                        비밀번호를 입력해주세요
                    </p>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <input 
                        type="password" 
                        id="password-input" 
                        placeholder="암호"
                        style="
                            flex: 1;
                            padding: 0.6rem;
                            border: 1px solid #ddd;
                            border-radius: 6px;
                            font-size: 0.9rem;
                            box-sizing: border-box;
                            outline: none;
                            transition: border-color 0.2s;
                        "
                    />
                    <button 
                        id="password-submit-btn"
                        style="
                            padding: 0.6rem 1rem;
                            background: #007bff;
                            color: white;
                            border: none;
                            border-radius: 6px;
                            font-size: 0.9rem;
                            cursor: pointer;
                            transition: background 0.2s;
                        "
                        onmouseover="this.style.background='#0056b3'"
                        onmouseout="this.style.background='#007bff'"
                    >
                        확인
                    </button>
                </div>
                <div id="password-error" style="
                    color: #dc3545;
                    margin-top: 0.8rem;
                    display: none;
                    font-size: 0.8rem;
                    text-align: center;
                ">
                    ❌ 잘못된 암호입니다
                </div>
            </div>
        `;
        
        document.body.appendChild(passwordDiv);
        console.log('✅ Password modal added to body');
        
        // 이벤트 리스너 추가
        const input = document.getElementById('password-input');
        const submitBtn = document.getElementById('password-submit-btn');
        
        if (input && submitBtn) {
            console.log('✅ Found input and submit button');
            // 버튼 클릭 이벤트
            submitBtn.addEventListener('click', validateLifePassword);
            
            // Enter 키 이벤트
            input.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    validateLifePassword();
                }
            });
            
            // 포커스
            setTimeout(() => {
                (input as HTMLInputElement).focus();
                console.log('✅ Input focused');
            }, 100);
        } else {
            console.log('❌ Could not find input or submit button');
        }
        
        console.log('🔒 Password prompt created and shown');
    }
    
    // 암호 검증 함수
    function validateLifePassword() {
        const input = document.getElementById('password-input') as HTMLInputElement;
        const error = document.getElementById('password-error');
        
        if (!input) return;
        
        if (input.value === '0508') {
            sessionStorage.setItem('life-auth', 'granted');
            removePasswordProtection();
            console.log('✅ Password correct, access granted');
        } else {
            // 암호 틀림
            if (error) {
                error.style.display = 'block';
            }
            input.value = '';
            input.focus();
            console.log('❌ Incorrect password');
        }
    }
    
    // 전역 함수로 등록
    (window as any).validateLifePassword = validateLifePassword;
    
    // 즉시 실행 (스크립트 로드와 함께)
    console.log('🚀 Life password script loaded, immediate check...');
    checkAndProtectLifePage();
    
    // 조금 후 재실행 (DOM 요소가 준비될 시간 확보)
    setTimeout(() => {
        console.log('⏱️ Delayed check after script load...');
        checkAndProtectLifePage();
    }, 10);
    
    // 더 늦은 시점에도 실행
    setTimeout(() => {
        console.log('⏱️ Second delayed check...');
        checkAndProtectLifePage();
    }, 100);
    
    // DOM이 완전히 로드된 후에도 실행
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            console.log('📄 DOM loaded, checking Life page...');
            setTimeout(checkAndProtectLifePage, 50);
        });
    } else {
        // DOM이 이미 로드된 경우 즉시 실행
        setTimeout(checkAndProtectLifePage, 50);
    }
    
    // 윈도우 로드 완료 후에도 실행 (이미지 등 모든 리소스 로드 후)
    window.addEventListener('load', () => {
        console.log('🌍 Window loaded, checking Life page...');
        setTimeout(checkAndProtectLifePage, 100);
    });
    
    // Quartz SPA 네비게이션 대응
    document.addEventListener('nav', () => {
        console.log('🔄 Navigation detected, checking Life page...');
        setTimeout(checkAndProtectLifePage, 100);
    });
    
    // 브라우저의 뒤로 가기/앞으로 가기 버튼 대응
    window.addEventListener('popstate', () => {
        console.log('⬅️ Popstate detected, checking Life page...');
        setTimeout(checkAndProtectLifePage, 100);
    });
    
    // URL 변경 감지 (pushState, replaceState 감지)
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;
    
    history.pushState = function(...args) {
        originalPushState.apply(this, args);
        console.log('📍 PushState detected, checking Life page...');
        setTimeout(checkAndProtectLifePage, 100);
    };
    
    history.replaceState = function(...args) {
        originalReplaceState.apply(this, args);
        console.log('🔄 ReplaceState detected, checking Life page...');
        setTimeout(checkAndProtectLifePage, 100);
    };
    
    // 페이지 내용 변경 감지 (MutationObserver)
    const observer = new MutationObserver(() => {
        // 너무 자주 실행되지 않도록 디바운스
        clearTimeout((window as any).lifePageCheckTimeout);
        (window as any).lifePageCheckTimeout = setTimeout(() => {
            console.log('🔍 DOM mutation detected, checking Life page...');
            checkAndProtectLifePage();
        }, 200);
    });
    
    // body의 변경사항 감지
    if (document.body) {
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    } else {
        // body가 아직 없으면 DOM 로드 후 시작
        document.addEventListener('DOMContentLoaded', () => {
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
    }
    
    // 지속적인 체크 (2초마다 - Life 페이지일 때만)
    checkInterval = setInterval(() => {
        const isLifePage = /\/[Ll]ife($|\/)/i.test(window.location.pathname);
        if (isLifePage) {
            console.log('⏰ Periodic check for Life page...');
            checkAndProtectLifePage();
        }
    }, 2000);
})();
