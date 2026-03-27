$outputDir = "..\datasets"

if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir
}

Get-ChildItem -Path $PWD -Filter "*.zip" | ForEach-Object {    
    Expand-Archive -Path $_.FullName -DestinationPath $outputDir -Force
}