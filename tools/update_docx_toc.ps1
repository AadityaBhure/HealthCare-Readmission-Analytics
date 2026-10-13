# Optional (Windows + Microsoft Word): fill in the DOCX table of contents and page references.
# Without this step Word offers to update the fields when the document is opened.
#   powershell -ExecutionPolicy Bypass -File tools\update_docx_toc.ps1
$docx = Join-Path $PSScriptRoot "..\reports\Project_Report.docx" | Resolve-Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx.Path, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Save()
    Write-Output ("Updated table of contents: " + $docx.Path + " (" + $doc.ComputeStatistics(2) + " pages)")
    $doc.Close($false)
} finally {
    $word.Quit()
}
