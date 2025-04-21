const fs = require("fs")
const path = require("path")

const contentDir = path.join(__dirname, "content")

function walk(dir, callback) {
  fs.readdirSync(dir, { withFileTypes: true }).forEach(entry => {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      walk(fullPath, callback)
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      callback(fullPath)
    }
  })
}

function convertImageLinks(filePath) {
  let content = fs.readFileSync(filePath, "utf8")
  const before = content

  // ![[IMG-20250421202904.png]] → ![](/img/IMG-20250421202904.png)
  content = content.replace(/!\[\[([^\]]+\.(?:png|jpe?g|gif|svg))\]\]/gi, (_, filename) => {
    return `![](/img/${filename})`
  })

  if (content !== before) {
    fs.writeFileSync(filePath, content, "utf8")
    console.log(`✔ Updated: ${filePath}`)
  }
}

walk(contentDir, convertImageLinks)
