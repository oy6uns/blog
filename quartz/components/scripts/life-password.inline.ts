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
        
        // 콘텐츠 표시
        const elements = document.querySelectorAll('body > *:not(#password-protection)');
        elements.forEach(el => {
            if ((el as HTMLElement).style.display === 'none') {
                (el as HTMLElement).style.display = '';
            }
        });
    }
    
    function showPasswordPrompt() {
        // 이미 암호 창이 있으면 리턴
        if (document.getElementById('password-protection')) return;
        
        passwordProtectionActive = true;
        
        // 페이지의 모든 콘텐츠 숨기기 (더 포괄적으로)
        const bodyChildren = document.querySelectorAll('body > *');
        bodyChildren.forEach(el => {
            if (el.tagName !== 'SCRIPT' && el.tagName !== 'STYLE') {
                (el as HTMLElement).style.display = 'none';
            }
        });
        
        // 암호 입력 폼 생성
        const passwordDiv = document.createElement('div');
        passwordDiv.id = 'password-protection';
        passwordDiv.innerHTML = `
            <div style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
            ">
                <div style="
                    background: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    max-width: 400px;
                    width: 90%;
                ">
                    <h2 style="margin-bottom: 1rem; color: #333;">🔒 보호된 콘텐츠</h2>
                    <p style="margin-bottom: 1.5rem; color: #666;">
                        이 페이지는 암호로 보호되어 있습니다.
                    </p>
                    <input 
                        type="password" 
                        id="password-input" 
                        placeholder="암호를 입력하세요"
                        style="
                            width: 100%;
                            padding: 0.75rem;
                            border: 1px solid #ddd;
                            border-radius: 4px;
                            margin-bottom: 1rem;
                            font-size: 1rem;
                            box-sizing: border-box;
                        "
                    />
                    <button 
                        id="password-submit-btn"
                        style="
                            width: 100%;
                            padding: 0.75rem;
                            background: #007bff;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            font-size: 1rem;
                            cursor: pointer;
                        "
                    >
                        확인
                    </button>
                    <div id="password-error" style="
                        color: #dc3545;
                        margin-top: 1rem;
                        display: none;
                    ">
                        잘못된 암호입니다.
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(passwordDiv);
        
        // 이벤트 리스너 추가
        const input = document.getElementById('password-input');
        const submitBtn = document.getElementById('password-submit-btn');
        
        if (input && submitBtn) {
            // 버튼 클릭 이벤트
            submitBtn.addEventListener('click', validateLifePassword);
            
            // Enter 키 이벤트
            input.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    validateLifePassword();
                }
            });
            
            // 포커스
            setTimeout(() => input.focus(), 100);
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
