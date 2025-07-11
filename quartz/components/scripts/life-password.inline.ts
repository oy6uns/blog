// Life 폴더 암호 보호
(function() {
    let passwordProtectionActive = false;
    
    function checkAndProtectLifePage() {
        const currentPath = window.location.pathname;
        console.log('🔍 Current path:', currentPath);
        
        // Life 폴더 경로 확인 (더 정확한 패턴 매칭)
        const isLifePage = /\/[Ll]ife($|\/)/i.test(currentPath);
        console.log('🔍 Is Life page:', isLifePage);
        
        if (!isLifePage) {
            console.log('❌ Not a Life page, skipping password protection');
            // Life 페이지가 아니면 기존 보호 제거
            removePasswordProtection();
            return;
        }
        
        console.log('✅ Life page detected!');
        
        // 이미 인증된 경우 스킵
        const hasAccess = sessionStorage.getItem('life-auth') === 'granted';
        console.log('🔍 Has access:', hasAccess);
        
        if (hasAccess) {
            console.log('✅ Already authenticated, skipping');
            removePasswordProtection();
            return;
        }
        
        if (passwordProtectionActive) {
            console.log('🔒 Password protection already active');
            return;
        }
        
        console.log('🔒 Need authentication, showing password prompt');
        showPasswordPrompt();
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
        console.log('🚀 showPasswordPrompt called');
        
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
    
    // 초기 실행
    checkAndProtectLifePage();
    
    // Quartz SPA 네비게이션 대응
    document.addEventListener('nav', () => {
        console.log('🔄 Navigation detected, checking Life page...');
        setTimeout(checkAndProtectLifePage, 100);
    });
})();
