var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var classNames = [];
var paint;
var dps = []
var model;

context = document.getElementById("canvas").getContext("2d");

$('#canvas').mousedown(function(e){
    var mouseX = e.pageX - this.offsetLeft;
    var mouseY = e.pageY - this.offsetTop;

    paint = true;
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
    redraw();
});

$('#canvas').mousemove(function(e){
    if(paint){
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
        redraw();
    }
});

$("#canvas").mouseup(function(e){
    paint = false;
    predictImage();
});

$('#canvas').mouseleave(function(e){
    paint = false;
});

$('#clearbutton').mousedown(function(e){
    clearCanvas()
})

function addClick(x, y, dragging)
{
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

function clearCanvas(){
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
}

function redraw(){
    context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
    
    context.strokeStyle = "#696969"
    context.lineJoin = "round6";
    context.lineWidth = 5;
              
    for(var i=0; i < clickX.length; i++) {		
      context.beginPath();
      if(clickDrag[i] && i){
        context.moveTo(clickX[i-1], clickY[i-1]);
       }else{
         context.moveTo(clickX[i]-1, clickY[i]);
       }
       context.lineTo(clickX[i], clickY[i]);
       context.closePath();
       context.stroke();
    }
}

async function loadClasses(){
    await $.ajax({
        url : "model/classes.txt",
        dataType : 'text',
    }).done(function(data){
        const lst = data.split(/\r\n/);
        for(var i = 0; i<lst.length - 1; i++ ){
            let s = lst[i];
            classNames[i] = s;
        } 
    });
}


function predictImage(){

    if(clickX.length >= 2){

        var img = getImage();
        var preprocessed_image = preprocessing(img);
        var pred = model.predict(preprocessed_image).dataSync();
        const indices = getIndexOfTop(pred, 4);
        const topValues = getTopValues(pred, 4);
        const names = getClassNames(indices);
        setTopPrediction(names, topValues);
        loadChart(names, topValues);

    }
}

function getImage(){    
    var minx = Math.min.apply(null,clickX);
    var miny = Math.min.apply(null,clickY);
    var maxx = Math.max.apply(null,clickX);
    var maxy = Math.max.apply(null,clickY);

    const dpi = window.devicePixelRatio;
    const imgData = context.getImageData(minx * dpi, miny * dpi, (maxx - minx)*dpi, (maxy - miny)*dpi);

    return imgData;
}


function preprocessing(imgData){
        let tensor = tf.browser.fromPixels(imgData, numChannels = 1);
        var resized_image = tf.image.resizeBilinear(tensor, [28,28]).toFloat();
        var normalised = resized_image.div(tf.scalar(105.0));
        var final_img = normalised.expandDims(0);
        return final_img;
}

async function start(){
    model = await tf.loadLayersModel('model/model.json');
    console.log(model.predict(tf.zeros([1, 28, 28, 1])).print());
    await loadClasses();
    let status = document.getElementById("status");
    status.innerHTML = "Model Loaded";
}


// Functions to get Result Data

function getIndexOfTop(data, num){
    var indices = [];
    var output = []
    for (var i = 0; i < data.length; i++) {
        indices.push(i); 
        if (indices.length > num) {
            indices.sort(function(a, b) {return data[b] - data[a]; }); 
            indices.pop();         }
    }
    return indices;
}

function getTopValues(data, num){
    var topValues = data.sort(function(a,b) {return b-a;}).slice(0,num);
    return topValues;
}

function getClassNames(indices){
    var outp = []
    for(var i=0; i < indices.length; i++){
        outp[i] = classNames[indices[i]];
    }

    return outp;
}


function loadChart(names, values) {
    dps = [];
    for(var i =0; i< 4; i++){
        dps.push({y : Math.round((values[i] * 100)*100)/100, label: names[i]});
    }

    console.log(dps);

    var chart = new CanvasJS.Chart("chartContainer", {
        animationEnabled: true,
        theme: "light2", // "light1", "light2", "dark1", "dark2"
        title:{
            text: "Predictions"
        },
        axisY: {
            title: "Percentage"
        },
        axisX:{
            title : "Class Name"
        },
        data: [{        
            type: "column",  
            dataPoints: dps
        }]
    });
    chart.render();   
}

function setTopPrediction(names, values){
    document.querySelector("#pred").innerHTML = names[0].toUpperCase() + " " + Math.round((values[0] * 100)*100)/100 + "%";
}

start();