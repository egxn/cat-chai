const express = require('express')
const bodyParser = require('body-parser')
const tf = require('@tensorflow/tfjs-node')
const Buffer = require('buffer').Buffer

const app = express()
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }))

let catchaiModel;
const labels = [`🐜`, `🦇`, `🐻`, `🐝`, `🐦`, `🦋`, `🐱` ]

async function loadModel() {
  if (!catchaiModel) {
    catchaiModel = await tf.node.loadSavedModel(__dirname + '/model/catchai')
  }
}

app.get('/', function (req, res) {
  res.sendFile('app/index.html', { root: __dirname })
})

app.get('/catchai', function (req, res) {
  res.json(labels[Math.floor(Math.random() * labels.length)])
})

app.post('/catchai', function (req, res) {
  const { animal, canvas } = req.body;
  const x0 = Buffer.from(canvas.replace(/^data:image\/(png|jpg);base64,/, ''), 'base64')
  const x1 = tf.node.decodePng(x0, 1)
  const x2 = tf.maxPool(x1, [4, 4], 4, "valid")
  const x3 = tf.cast(x2, 'float32')
  const x4 = x3.div(tf.scalar(255))
  const x5 = tf.reshape(x4, [-1, 28, 28, 1])
  const y = catchaiModel.predict(x5)
  const yValues = Array.from(y.dataSync())
  const max = Math.max(...yValues)
  res.json({
    drawed: labels[yValues.indexOf(max)],
    catchai: animal
  })
})

loadModel();

app.listen(3000, function () {
  console.log('Up on 3000 port!')
})