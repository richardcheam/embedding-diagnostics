/* Measured figures and discrete audit states. No interpolated research results. */
(() => {
  'use strict';
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const compact = window.matchMedia('(max-width: 700px)');
  const stage = document.querySelector('.audit-stage');
  const steps = [...document.querySelectorAll('.audit-step')];
  const buttons = [...document.querySelectorAll('[data-step]')];
  const notes = [
    'Two protocols. One encoder. The original score is retained as an audit record.',
    'The original optimizer settings underfit small-magnitude features. Test the instrument.',
    'Corrected fitting changes the reading. The encoder and evaluation split remain fixed.',
    'Label prediction and neighbour quality are separate endpoints. The Results section reports both.'
  ];
  function setStage(index) {
    stage.dataset.state = steps[index].dataset.state;
    document.getElementById('stage-count').textContent = `0${index + 1} / 04`;
    document.getElementById('stage-note').textContent = notes[index];
    buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
  }
  let observer;
  function configureMotion() {
    const enabled = !reduced.matches && !compact.matches;
    document.documentElement.classList.toggle('motion-ready', enabled);
    observer?.disconnect();
    if (!enabled) return;
    observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) setStage(steps.indexOf(entry.target));
      });
    }, { rootMargin: '-35% 0px -35% 0px', threshold: 0 });
    steps.forEach(step => observer.observe(step));
  }
  configureMotion();
  reduced.addEventListener('change', configureMotion);
  compact.addEventListener('change', configureMotion);
  buttons.forEach(button => button.addEventListener('click', () => {
    const index = Number(button.dataset.step);
    setStage(index);
    steps[index].scrollIntoView({ block: 'center', behavior: reduced.matches ? 'instant' : 'smooth' });
  }));

  const order = ['ema_stopgrad', 'none_stopgrad', 'none_nostopgrad'];
  const names = ['EMA reference', 'Stop-gradient only', 'Contracted control'];
  const palette = getComputedStyle(document.documentElement);
  const token = name => palette.getPropertyValue(`--${name}`).trim();
  const colors = ['reference', 'comparison', 'control'].map(token);
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
  const ns = 'http://www.w3.org/2000/svg';
  function svgElement(name, attributes, text) {
    const node = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function chart(data, key, title, logarithmic = false, floorKey = null) {
    const figure = document.createElement('figure');
    figure.className = 'metric-chart';
    const heading = document.createElement('h3');
    heading.textContent = title;
    figure.append(heading);
    const svg = svgElement('svg', {viewBox:'0 0 580 240', role:'img', 'aria-label':`${title}: five individual seeds and a mean marker per condition. Circle: EMA reference. Diamond: stop-gradient only. Square: contracted control.`});
    const left = 140, right = 495, top = 40, row = 51;
    const all = order.flatMap(condition => data.conditions[condition][key]);
    const low = logarithmic ? Math.floor(Math.log10(Math.min(...all))) : 0;
    const high = logarithmic ? Math.ceil(Math.log10(Math.max(...all))) : 1;
    const x = value => left + ((logarithmic ? Math.log10(value) : value) - low) / (high - low) * (right - left);
    const ticks = logarithmic ? Array.from({length:high - low + 1}, (_, i) => low + i) : [0, .25, .5, .75, 1];
    ticks.forEach(tick => {
      const position = logarithmic ? x(10 ** tick) : x(tick);
      svg.append(svgElement('line',{x1:position,y1:top-12,x2:position,y2:top+2*row+15,stroke:token('grid'),'stroke-width':1}));
      svg.append(svgElement('text',{x:position,y:top+2*row+39,'text-anchor':'middle',fill:token('muted'),'font-size':12,'font-family':'monospace'}, logarithmic ? `10^${tick}` : `${tick*100}%`));
    });
    if (floorKey) {
      const floor = mean(data.conditions[order[0]][floorKey]);
      svg.append(svgElement('line',{x1:x(floor),y1:top-15,x2:x(floor),y2:top+2*row+15,stroke:token('muted'),'stroke-dasharray':'4 4'}));
      svg.append(svgElement('text',{x:x(floor),y:18,'text-anchor':'middle',fill:token('muted'),'font-size':11,'font-family':'monospace'},`floor ${(floor*100).toFixed(1)}%`));
    }
    order.forEach((condition, i) => {
      const values = data.conditions[condition][key], y = top + i*row;
      svg.append(svgElement('text',{x:0,y:y+4,fill:colors[i],'font-size':11,'font-family':'Arial'},names[i]));
      values.forEach((value, seed) => {
        const px = x(value), py = y+(seed-2)*3;
        const shape = i === 0 ? ['circle', {cx:px,cy:py,r:4}]
          : i === 1 ? ['path', {d:`M${px},${py-5} l5,5 l-5,5 l-5,-5 Z`}]
          : ['rect', {x:px-4,y:py-4,width:8,height:8}];
        const dot = svgElement(shape[0],{...shape[1],fill:colors[i],class:'chart-dot'});
        dot.append(svgElement('title',{},`Seed ${seed}: ${logarithmic ? value.toPrecision(5) : (value*100).toFixed(3)+'%'}`));
        svg.append(dot);
      });
      const average = mean(values);
      svg.append(svgElement('line',{x1:x(average),x2:x(average),y1:y-13,y2:y+13,stroke:colors[i],class:'chart-mean'}));
      svg.append(svgElement('text',{x:right+10,y:y+4,fill:colors[i],'font-size':10,'font-family':'monospace'},logarithmic ? average.toPrecision(3) : (average*100).toFixed(1)+'%'));
    });
    figure.append(svg);
    const caption = document.createElement('figcaption');
    caption.textContent = `[ours] ${logarithmic ? 'Log axis.' : 'Linear axis.'} Symbols: seeds 0–4. Vertical marker: mean. Values at right: means.`;
    figure.append(caption);
    return figure;
  }
  function render(data) {
    const attribute = document.getElementById('attribute').value;
    const charts = document.getElementById('measured-charts');
    charts.replaceChildren(
      chart(data,'total_variance','Embedding spread / total variance',true),
      chart(data,`probe_accuracy_unscaled_${attribute}`,'Label prediction / corrected probe accuracy',false,`probe_majority_${attribute}`),
      chart(data,`probe_balanced_accuracy_unscaled_${attribute}`,'Label prediction / balanced accuracy'),
      chart(data,`retrieval_p10_${attribute}`,'Neighbour label agreement / P@10',false,`retrieval_chance_${attribute}`)
    );
    const table = document.createElement('table');
    const caption = document.createElement('caption');
    caption.textContent = `[ours] ${attribute} · corrected raw probe and retrieval · paired differences A − B`;
    table.append(caption);
    const head = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Contrast','Endpoint','Mean difference','95% interval'].forEach(label => {
      const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = label; headerRow.append(cell);
    });
    head.append(headerRow); table.append(head);
    const body = document.createElement('tbody');
    data.intervals.filter(item => [`probe_accuracy_unscaled_${attribute}`,`retrieval_p10_${attribute}`].includes(item.metric)).forEach(item => {
      const row = document.createElement('tr');
      const values = [item.name, item.metric.startsWith('probe') ? 'Raw probe' : 'P@10', `${(item.mean*100).toFixed(2)} pp`, `[${(item.low*100).toFixed(2)}, ${(item.high*100).toFixed(2)}] pp`];
      values.forEach((value,i) => { const cell=document.createElement(i===0?'th':'td'); if(i===0)cell.scope='row'; cell.textContent=value; row.append(cell); });
      body.append(row);
    });
    table.append(body);
    document.getElementById('interval-table').replaceChildren(table);
  }
  fetch('data.json').then(response => {
    if (!response.ok) throw new Error('Missing endpoint data');
    return response.json();
  }).then(data => {
    render(data);
    document.getElementById('attribute').addEventListener('change', () => render(data));
  }).catch(() => {document.getElementById('data-error').hidden=false;});
})();
