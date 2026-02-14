Add-Type -AssemblyName System.Drawing
$inputPath = "c:\Users\Falker\Desktop\Code\KT\paper\optical_aliasing_zoom.png"
$outputPath = "c:\Users\Falker\Desktop\Code\KT\paper\optical_aliasing_zoom_small.png"

$img = [System.Drawing.Image]::FromFile($inputPath)
$newWidth = [int]($img.Width * 0.5)
$newHeight = [int]($img.Height * 0.5)

$bmp = New-Object System.Drawing.Bitmap($newWidth, $newHeight)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g.DrawImage($img, 0, 0, $newWidth, $newHeight)

$img.Dispose()
$g.Dispose()

$bmp.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Write-Host "Resizing complete."
$oldSize = (Get-Item $inputPath).Length
$newSize = (Get-Item $outputPath).Length
Write-Host "Old size: $oldSize bytes"
Write-Host "New size: $newSize bytes"
