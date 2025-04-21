import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "oy6uns.dev",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "quartz.jzhao.xyz",
    ignorePatterns: ["private", "templates", ".obsidian", "**/assets/**"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Inter",
        body: "Inter",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light:     "#ffffff",  // bg
          lightgray: "#f1f5f9",  // Slate‑100
          gray:      "#6b7280",  // Gray‑500
          darkgray:  "#374151",  // Gray‑700
          dark:      "#111827",  // Gray‑900 (header/icons)
          secondary: "#3b82f6",  // Blue‑500  ← 링크·버튼
          tertiary:  "#6366f1",  // Indigo‑500 ← hover/visited
          highlight: "rgba(59,130,246,0.08)", // Blue‑500 @ 8 %
          textHighlight: "#fde04788",         // Amber‑300 @ 50 %

          // 옵시디언 기본
          // light: "#faf8f8",
          // lightgray: "#e5e5e5",
          // gray: "#b8b8b8",
          // darkgray: "#4e4e4e",
          // dark: "#2b2b2b",
          // secondary: "#284b63",
          // tertiary: "#84a59d",
          // highlight: "rgba(143, 159, 169, 0.15)",
          // textHighlight: "#fff23688",
        },
        darkMode: {
          light:     "#1f2937",  // Gray‑800
          lightgray: "#374151",  // Gray‑700
          gray:      "#9ca3af",  // Gray‑400
          darkgray:  "#d1d5db",  // Gray‑300
          dark:      "#f3f4f6",  // Gray‑100 (글자)
          secondary: "#60a5fa",  // Blue‑400
          tertiary:  "#a5b4fc",  // Indigo‑300
          highlight: "rgba(59,130,246,0.15)", // Blue‑500 @ 15 %
          textHighlight: "#fbbf2488",         // Amber‑400 @ 50 %
          // 옵시디언 기본
          // light: "#161618",
          // lightgray: "#393639",
          // gray: "#646464",
          // darkgray: "#d4d4d4",
          // dark: "#ebebec",
          // secondary: "#7b97aa",
          // tertiary: "#84a59d",
          // highlight: "rgba(143, 159, 169, 0.15)",
          // textHighlight: "#b3aa0288",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
