// Life 폴더 페이지에서만 암호 보호 JavaScript 로드
(function() {
    // Life 폴더 관련 페이지에서만 실행
    const currentPath = window.location.pathname;
    const isLifePage = currentPath.toLowerCase().includes('/life/') || 
                      document.querySelector('a[href*="/life/"], a[href*="/Life/"]');
    
    // Life와 관련 없는 페이지는 스킵
    if (!isLifePage && !currentPath.toLowerCase().includes('/life/')) {
        return;
    }

    // 이미 로드된 경우 중복 방지
    if (window.passwordProtectionLoaded) return;
    window.passwordProtectionLoaded = true;

    // 암호 보호된 콘텐츠 접근을 위한 JavaScript
    class PasswordProtection {
        constructor() {
            this.protectedPaths = ['/Life/', '/life/'];
            this.password = '0000'; // 실제 사용할 암호로 변경하세요
            this.sessionKey = 'quartz-auth-life';
            this.init();
        }

        init() {
            // 현재 페이지가 보호된 경로인지 확인
            const currentPath = window.location.pathname;
            const isProtected = this.protectedPaths.some(path => 
                currentPath.toLowerCase().includes(path.toLowerCase())
            );

            if (isProtected) {
                this.checkAccess();
            }

            // Life 폴더로의 링크 클릭 시 인터셉트
            this.interceptLinks();
        }

        checkAccess() {
            const hasAccess = sessionStorage.getItem(this.sessionKey) === 'granted';
            
            if (!hasAccess) {
                this.showPasswordPrompt();
            }
        }

        showPasswordPrompt() {
            // 페이지 콘텐츠 숨기기
            const content = document.querySelector('article, main, #quartz-body');
            if (content) {
                content.style.display = 'none';
            }

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
                            onclick="window.passwordProtection.validatePassword()"
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

            // Enter 키로 확인
            document.getElementById('password-input').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.validatePassword();
                }
            });

            // 입력 필드에 포커스
            document.getElementById('password-input').focus();
        }

        validatePassword() {
            const input = document.getElementById('password-input');
            const error = document.getElementById('password-error');
            
            if (input.value === this.password) {
                // 암호 맞음
                sessionStorage.setItem(this.sessionKey, 'granted');
                document.getElementById('password-protection').remove();
                
                // 콘텐츠 표시
                const content = document.querySelector('article, main, #quartz-body');
                if (content) {
                    content.style.display = '';
                }
            } else {
                // 암호 틀림
                error.style.display = 'block';
                input.value = '';
                input.focus();
            }
        }

        interceptLinks() {
            document.addEventListener('click', (e) => {
                const link = e.target.closest('a');
                if (link && link.href) {
                    const isProtectedLink = this.protectedPaths.some(path => 
                        link.href.toLowerCase().includes(path.toLowerCase())
                    );
                    
                    if (isProtectedLink) {
                        const hasAccess = sessionStorage.getItem(this.sessionKey) === 'granted';
                        if (!hasAccess) {
                            e.preventDefault();
                            this.showPasswordPrompt();
                        }
                    }
                }
            });
        }
    }

    // 페이지 로드 시 초기화
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.passwordProtection = new PasswordProtection();
        });
    } else {
        window.passwordProtection = new PasswordProtection();
    }
})();
