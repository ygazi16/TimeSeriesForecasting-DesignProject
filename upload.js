const path = "file:///C:/Users/ASUS/OneDrive/Desktop/upload/main.html";
var fs = require('fs');
var parse = require('csv-parse');



var csvData= document.getElementById("formFileLg");
var button = document.getElementById("clickbutton");


button.onclick = function(){
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

const main = document.querySelector("#main");

main.addEventListener("click", () => {
  window.location.href =  "file:///C:/Users/ASUS/OneDrive/Desktop/upload/main.html";
  });
