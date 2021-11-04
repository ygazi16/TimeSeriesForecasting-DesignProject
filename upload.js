const path = "file:///C:/Users/ASUS/OneDrive/Desktop/upload/main.html";
var fs = require('fs');
var parse = require('csv-parse');



var csvData= document.getElementById("formFileLg");
var button1 = document.getElementById("clickbutton1");
var button2 = document.getElementById("clickbutton2");

button1.onclick = function(){
  fs.createReadStream(req.file.path)
      .pipe(parse({delimiter: ':'}))
      .on('data', function(csvrow) {
          console.log(csvrow);
          //do something with csvrow
          csvData.push(csvrow);
      })
      .on('end',function() {
        //do something with csvData
        console.print(csvData);
      });
}

button2.onclick() = function saveTextAsFile()
{
    var textToWrite = document.getElementById("inputTextToSave").value;
    var textFileAsBlob = new Blob([textToWrite], {type:'text/plain'});
    var fileNameToSaveAs = document.getElementById("inputFileNameToSaveAs").value;
      var downloadLink = document.createElement("a");
    downloadLink.download = fileNameToSaveAs;
    downloadLink.innerHTML = "Download File";
    if (window.webkitURL != null)
    {
        // Chrome allows the link to be clicked
        // without actually adding it to the DOM.
        downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
    }
    else
    {
        // Firefox requires the link to be added to the DOM
        // before it can be clicked.
        downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
        downloadLink.onclick = destroyClickedElement;
        downloadLink.style.display = "none";
        document.body.appendChild(downloadLink);
    }

    downloadLink.click();
}

const main = document.querySelector("#main");

main.addEventListener("click", () => {
  window.location.href =  "file:///C:/Users/ASUS/OneDrive/Desktop/upload/main.html";
  });
