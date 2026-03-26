Get-ChildItem -Path "..\datasets" -Directory | ForEach-Object {
    $zipPath = Join-Path $PWD -ChildPath "$($_.Name).zip"
    Write-Host "Compressing $($_.Name) to $($_.Name).zip"
    Compress-Archive -Path $_.FullName -DestinationPath $zipPath -Force
}