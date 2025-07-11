#!/bin/bash

# Archive/Tips 폴더의 마크다운 파일들에 날짜 추가
# Git 히스토리가 없는 파일들은 현재 날짜로 설정

echo "📝 Archive/Tips 폴더 파일들의 날짜 추가를 시작합니다..."

# Archive/Tips 디렉토리의 모든 .md 파일을 찾습니다
find content/Archive/Tips -name "*.md" -type f | while read file; do
    echo "처리 중: $file"
    
    # 파일의 첫 번째 커밋 날짜 찾기 (생성일)
    created_date=$(git log --follow --format="%ad" --date=short -- "$file" 2>/dev/null | tail -1)
    
    # 파일의 마지막 커밋 날짜 찾기 (수정일)
    modified_date=$(git log --follow --format="%ad" --date=short -- "$file" 2>/dev/null | head -1)
    
    # Git 히스토리가 없는 경우 현재 날짜 사용
    if [ -z "$created_date" ]; then
        created_date=$(date +%Y-%m-%d)
        modified_date=$(date +%Y-%m-%d)
        echo "  ⚠️  Git 히스토리가 없어 현재 날짜로 설정합니다."
    fi
    
    echo "  생성일: $created_date"
    echo "  수정일: $modified_date"
    
    # 임시 파일 생성
    temp_file=$(mktemp)
    
    # 파일에 frontmatter가 있는지 확인
    if head -1 "$file" | grep -q "^---"; then
        echo "  📋 기존 frontmatter가 있습니다 - 날짜 정보 추가/수정"
        # 기존 frontmatter가 있는 경우 - 날짜 정보 추가/수정
        awk -v created="$created_date" -v modified="$modified_date" '
        BEGIN { in_frontmatter = 0; date_added = 0; created_added = 0; modified_added = 0 }
        /^---$/ { 
            if (in_frontmatter == 0) {
                in_frontmatter = 1
                print $0
            } else {
                # frontmatter 끝나기 전에 누락된 필드들 추가
                if (date_added == 0) {
                    print "date: " created
                }
                if (created_added == 0) {
                    print "created: " created
                }
                if (modified_added == 0) {
                    print "modified: " modified
                }
                print $0
                in_frontmatter = 0
            }
            next
        }
        /^date:/ && in_frontmatter == 1 { 
            print "date: " created
            date_added = 1
            next 
        }
        /^created:/ && in_frontmatter == 1 { 
            print "created: " created
            created_added = 1
            next 
        }
        /^modified:/ && in_frontmatter == 1 { 
            print "modified: " modified
            modified_added = 1
            next 
        }
        { print $0 }
        ' "$file" > "$temp_file"
    else
        echo "  📄 frontmatter가 없습니다 - 새로 추가"
        # frontmatter가 없는 경우 - 새로 추가
        {
            echo "---"
            echo "date: $created_date"
            echo "created: $created_date"
            echo "modified: $modified_date"
            echo "tags: []"
            echo "---"
            echo ""
            cat "$file"
        } > "$temp_file"
    fi
    
    # 원본 파일 교체
    mv "$temp_file" "$file"
    echo "  ✅ 날짜 정보가 추가되었습니다."
    echo ""
done

echo "🎉 Archive/Tips 폴더 날짜 추가가 완료되었습니다!"
echo "📋 변경사항을 확인하고 필요시 수정하세요."
