let RunVideoAnalysis = () =>{
    urlToAnalyze = document.getElementById("urlToAnalyze").value;

    let xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState === 4 && this.status === 200) {
            document.getElementById("system_response_video").innerHTML = xhttp.responseText;
        }
    };
    xhttp.open("GET", "videoDetector?urlToAnalyze="+urlToAnalyze, true);
    xhttp.send();
}