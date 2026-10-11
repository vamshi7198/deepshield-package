# Renders every slide of a .pptx to PNG using the installed PowerPoint (for visual QA).
# Usage: powershell -ExecutionPolicy Bypass -File render_pptx.ps1 <deck.pptx> <outDir> [widthPx]
param([string]$Deck, [string]$OutDir, [int]$Width = 1600)
$ErrorActionPreference = "Stop"
$Deck = (Resolve-Path $Deck).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
Get-ChildItem $OutDir -Filter "slide-*.png" -ErrorAction SilentlyContinue | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($Deck, $true, $false, $false)   # read-only, no window
    $ratio = $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth
    $h = [int]($Width * $ratio)
    foreach ($s in $pres.Slides) {
        $name = "slide-{0:D2}.png" -f $s.SlideIndex
        $s.Export((Join-Path $OutDir $name), "PNG", $Width, $h)
    }
    $pres.Close()
    Write-Output "Rendered $($pres.Slides.Count) slides to $OutDir"
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
