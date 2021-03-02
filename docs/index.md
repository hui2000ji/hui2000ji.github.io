# Huiyu CAI 蔡辉宇

![profile photo](assets/images/photo.jpg){: align=right style="width: 25%" }

I am a fourth-year undergraduate at the <a href="http://www.cis.pku.edu.cn/">Department of Machine Intelligence</a>,
<a href="http://www.pku.edu.cn">Peking University</a>.
I was previously a member of the Language Computing and Web Mining Group,
led by <a href="https://wanxiaojun.github.io/">Prof. Xiaojun Wan</a>.
This autumn, I will continue my studies at <a href="https://mila.quebec/en">MILA</a>
under the supervision of <a href="https://jian-tang.com">Jian Tang</a>.

I am interested in deep generative models, graph representation learning and their wide applications,
such as drug discovery, single-cell modeling, etc.

## Selected Publications

![scETM](assets/images/scETM.png){: align=right style="width: 25%" }

- [**Learning interpretable cellular and gene signature embeddings from single-cell transcriptomic data**](https://www.biorxiv.org/content/10.1101/2021.01.13.426593v1.full)<br><em>bioRxiv</em>, 2020 &emsp;[code](https://github.com/hui2000ji/scETM)<br><a href="https://yifnzhao.github.io/">Yifan Zhao</a>\*, <strong>Huiyu Cai</strong>\*, <a href="https://oxer11.github.io/">Zuobai Zhang</a>, <a href="https://jian-tang.com">Jian Tang</a>, <a href="https://www.cs.mcgill.ca/~yueli/">Yue Li</a><br> *<em>Equal contribution</em>

![multi-modal sarcasm detection](assets/images/ACL_2019.jpg){: align=right style="width: 25%" }

- [**Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model**](https://www.aclweb.org/anthology/P19-1239/) <br><em>ACL</em>, 2019 &emsp;[data](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)<br>Yitao Cai, <strong>Huiyu Cai</strong>, <a href="https://wanxiaojun.github.io/">Xiaojun Wan</a>

## For-Fun Projects
- [**Implementation of the EWLS Algorithm for the Maximum Clique Problem**](assets/codes/EWLS.cpp) (Dec. 2020 - Jan. 2021)
- [**Music Source Separation: Theory and Applications**](assets/documents/Music%20Source%20Separation%20-%20Report.pdf) (Apr. 2020 - Jun. 2020)
- [**Raiden Game Implementation in Java**](https://github.com/hui2000ji/RaidenGame) (Jan. 2020 - Jun. 2020)
- [**Fine-grained Face Manipulation via DLGAN**](https://github.com/sunyaofeng8/AI-Intro) (Oct. 2019 - Jan. 2020)
- **Bird Sound Classification via CNN** (Mar. 2019 - Jun. 2019)
- **[Mahjong](https://www.botzone.org.cn/game/Mahjong-GB) AI Based on Supervised Learning** (Mar. 2019 - Jun. 2019)
- **Rule-based [Doudizhu](https://www.botzone.org.cn/game/FightTheLandlord2) & [Ataxx](https://www.botzone.org.cn/game/Ataxx) & [Reversi](https://www.botzone.org.cn/game/Reversi) & [pysc2-minimap](https://github.com/deepmind/pysc2) AI** (Oct. 2017 - May. 2019)

## Change Color Palette

Click on a tile to change the color scheme:

<div class="tx-switch">
  <button data-md-color-scheme="default"><code>default</code></button>
  <button data-md-color-scheme="slate"><code>slate</code></button>
</div>

<script>
  var buttons = document.querySelectorAll("button[data-md-color-scheme]")
  buttons.forEach(function(button) {
    button.addEventListener("click", function() {
      var attr = this.getAttribute("data-md-color-scheme")
      document.body.dataset.mdColorScheme = attr
      localStorage.setItem("data-md-color-scheme", attr);

    })
  })
</script>

Click on a tile to change the primary color:

<style>
  .md-typeset button[data-md-color-primary] > code {
    background-color: var(--md-primary-fg-color);
    color: var(--md-primary-bg-color);
  }
</style>

<div class="tx-switch">
  <button data-md-color-primary="red"><code>red</code></button>
  <button data-md-color-primary="pink"><code>pink</code></button>
  <button data-md-color-primary="purple"><code>purple</code></button>
  <button data-md-color-primary="deep-purple"><code>deep purple</code></button>
  <button data-md-color-primary="indigo"><code>indigo</code></button>
  <button data-md-color-primary="blue"><code>blue</code></button>
  <button data-md-color-primary="light-blue"><code>light blue</code></button>
  <button data-md-color-primary="cyan"><code>cyan</code></button>
  <button data-md-color-primary="teal"><code>teal</code></button>
  <button data-md-color-primary="green"><code>green</code></button>
  <button data-md-color-primary="light-green"><code>light green</code></button>
  <button data-md-color-primary="lime"><code>lime</code></button>
  <button data-md-color-primary="yellow"><code>yellow</code></button>
  <button data-md-color-primary="amber"><code>amber</code></button>
  <button data-md-color-primary="orange"><code>orange</code></button>
  <button data-md-color-primary="deep-orange"><code>deep orange</code></button>
  <button data-md-color-primary="brown"><code>brown</code></button>
  <button data-md-color-primary="grey"><code>grey</code></button>
  <button data-md-color-primary="blue-grey"><code>blue grey</code></button>
  <button data-md-color-primary="black"><code>black</code></button>
  <button data-md-color-primary="white"><code>white</code></button>
</div>

<script>
  var buttons = document.querySelectorAll("button[data-md-color-primary]");
  Array.prototype.forEach.call(buttons, function(button) {
    button.addEventListener("click", function() {
      document.body.dataset.mdColorPrimary = this.dataset.mdColorPrimary;
      localStorage.setItem("data-md-color-primary", this.dataset.mdColorPrimary);
    })
  })
</script>

Click on a tile to change the accent color:

<style>
  .md-typeset button[data-md-color-accent] > code {
    background-color: var(--md-code-bg-color);
    color: var(--md-accent-fg-color);
  }
</style>

<div class="tx-switch">
  <button data-md-color-accent="red"><code>red</code></button>
  <button data-md-color-accent="pink"><code>pink</code></button>
  <button data-md-color-accent="purple"><code>purple</code></button>
  <button data-md-color-accent="deep-purple"><code>deep purple</code></button>
  <button data-md-color-accent="indigo"><code>indigo</code></button>
  <button data-md-color-accent="blue"><code>blue</code></button>
  <button data-md-color-accent="light-blue"><code>light blue</code></button>
  <button data-md-color-accent="cyan"><code>cyan</code></button>
  <button data-md-color-accent="teal"><code>teal</code></button>
  <button data-md-color-accent="green"><code>green</code></button>
  <button data-md-color-accent="light-green"><code>light green</code></button>
  <button data-md-color-accent="lime"><code>lime</code></button>
  <button data-md-color-accent="yellow"><code>yellow</code></button>
  <button data-md-color-accent="amber"><code>amber</code></button>
  <button data-md-color-accent="orange"><code>orange</code></button>
  <button data-md-color-accent="deep-orange"><code>deep orange</code></button>
</div>

<script>
  var buttons = document.querySelectorAll("button[data-md-color-accent]");
  Array.prototype.forEach.call(buttons, function(button) {
    button.addEventListener("click", function() {
      document.body.dataset.mdColorAccent = this.dataset.mdColorAccent;
      localStorage.setItem("data-md-color-accent", this.dataset.mdColorAccent);
    })
  })
  document.getElementsByClassName('md-nav__title')[1].click()
</script>