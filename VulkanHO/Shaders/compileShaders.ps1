$compilerPath = "D:/VulkanSDK/1.3.275.0/Bin/glslc.exe"
$extens = @("*.vert", "*.frag", "*.comp")

$files = Get-ChildItem -Recurse -Attributes Archive -Include $extens

if ($files.count -eq 0)
{
    Write-Host "Nothing to compile"
}
else
{
    foreach ($file in $files)
    {
        $file.Attributes = $file.Attributes -band (-bnot [System.IO.FileAttributes]::Archive)
        Invoke-Expression "$($compilerPath) $($file.FullName) -o $($file.FullName + ".spv")"
        Write-Host "Compiled " -NoNewline
        Write-Host $file.Name -NoNewline -ForegroundColor Red
        Write-Host " into " -NoNewline
        Write-Host $($file.Name + ".spv") -ForegroundColor Yellow
    }
}
Write-Host ""
Read-Host -Prompt "Press Enter to exit"