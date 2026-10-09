/* Measured final-checkpoint figures. */
(() => {
  'use strict';
  const order = ['ema_stopgrad', 'none_stopgrad', 'sigreg_stopgrad', 'sigreg_nostopgrad', 'proj_sigreg_stopgrad', 'proj_sigreg_nostopgrad', 'none_nostopgrad'];
  const names = ['EMA reference', 'Stop-gradient only', 'SIGReg + stop-gradient', 'SIGReg', 'Projector + SIGReg + stop-gradient', 'Projector + SIGReg', 'Contracted control'];
  const palette = getComputedStyle(document.documentElement);
  const token = name => palette.getPropertyValue(`--${name}`).trim();
  const colors = ['reference', 'comparison', 'ink', 'ink', 'ink', 'ink', 'control'].map(token);
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
  const ns = 'http://www.w3.org/2000/svg';
  function svgElement(name, attributes, text) {
    const node = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function chart(data, key, title, logarithmic = false, floorKey = null, compact = false) {
    const figure = document.createElement('figure');
    figure.className = 'metric-chart';
    const heading = document.createElement('h3');
    heading.textContent = title;
    figure.append(heading);
    const left = compact ? 25 : 225, right = compact ? 335 : 585;
    const top = compact ? 65 : 45, row = compact ? 75 : 49;
    const bottom = top + (order.length - 1) * row + 18;
    const svg = svgElement('svg', {viewBox:`0 0 ${compact ? 360 : 680} ${bottom+48}`, role:'img', 'aria-label':`${title}: all seven conditions, five individual seeds and a mean marker per condition. ${names.join('; ')}.`});
    const all = order.flatMap(condition => data.conditions[condition][key]);
    const low = logarithmic ? Math.floor(Math.log10(Math.min(...all))) : 0;
    const high = logarithmic ? Math.ceil(Math.log10(Math.max(...all))) : 1;
    const x = value => left + ((logarithmic ? Math.log10(value) : value) - low) / (high - low) * (right - left);
    const ticks = logarithmic ? Array.from({length:high - low + 1}, (_, i) => low + i) : [0, .25, .5, .75, 1];
    ticks.forEach(tick => {
      const position = logarithmic ? x(10 ** tick) : x(tick);
      svg.append(svgElement('line',{x1:position,y1:top-12,x2:position,y2:bottom,stroke:token('grid'),'stroke-width':1}));
      svg.append(svgElement('text',{x:position,y:bottom+26,'text-anchor':'middle',fill:token('muted'),'font-size':12,'font-family':'monospace'}, logarithmic ? `10^${tick}` : `${tick*100}%`));
    });
    if (floorKey) {
      const floor = mean(data.conditions[order[0]][floorKey]);
      svg.append(svgElement('line',{x1:x(floor),y1:top-15,x2:x(floor),y2:bottom,stroke:token('muted'),'stroke-dasharray':'4 4'}));
      svg.append(svgElement('text',{x:x(floor),y:18,'text-anchor':'middle',fill:token('muted'),'font-size':11,'font-family':'monospace'},`floor ${(floor*100).toFixed(1)}%`));
    }
    order.forEach((condition, i) => {
      const values = data.conditions[condition][key], y = top + i*row;
      const label = svgElement('text',{x:compact ? left : 0,y:compact ? y-30 : y+4,fill:colors[i],'font-size':compact ? 13 : 14,'font-family':'Arial',class:'chart-condition'});
      if (names[i] === 'Projector + SIGReg + stop-gradient') {
        label.append(svgElement('tspan',{x:compact ? left : 0,dy:compact ? 0 : -7},'Projector + SIGReg '));
        label.append(svgElement('tspan',{x:compact ? left : 0,dy:16},'+ stop-gradient'));
      } else label.textContent = names[i];
      svg.append(label);
      values.forEach((value, seed) => {
        const px = x(value), py = y+(seed-2)*3;
        const shape = condition === 'ema_stopgrad' ? ['circle', {cx:px,cy:py,r:4}]
          : condition === 'none_stopgrad' ? ['path', {d:`M${px},${py-5} l5,5 l-5,5 l-5,-5 Z`}]
          : condition === 'none_nostopgrad' ? ['rect', {x:px-4,y:py-4,width:8,height:8}]
          : ['circle', {cx:px,cy:py,r:4}];
        const dot = svgElement(shape[0],{...shape[1],fill:colors[i],class:'chart-dot'});
        dot.append(svgElement('title',{},`Seed ${seed}: ${logarithmic ? value.toPrecision(5) : (value*100).toFixed(3)+'%'}`));
        svg.append(dot);
      });
      const average = mean(values);
      svg.append(svgElement('line',{x1:x(average),x2:x(average),y1:y-13,y2:y+13,stroke:colors[i],class:'chart-mean'}));
      svg.append(svgElement('text',{x:compact ? right : right+12,y:compact ? y-30 : y+4,'text-anchor':compact ? 'end' : 'start',fill:colors[i],'font-size':compact ? 12 : 13,'font-family':'monospace'},logarithmic ? average.toPrecision(3) : (average*100).toFixed(2)+'%'));
    });
    figure.append(svg);
    const caption = document.createElement('figcaption');
    caption.textContent = `${logarithmic ? 'Log axis.' : 'Linear axis.'} All seven conditions. Symbols: seeds 0–4. Vertical marker: mean. Values at right: means.`;
    figure.append(caption);
    return figure;
  }
  let layout;
  function render(data) {
    const attribute = document.getElementById('attribute').value;
    const charts = document.getElementById('measured-charts');
    const columns = getComputedStyle(charts).gridTemplateColumns.split(' ').length;
    layout = `${charts.clientWidth}:${columns}`;
    const compact = charts.clientWidth / columns < 480;
    charts.replaceChildren(
      chart(data,'total_variance','Embedding spread / total variance',true,null,compact),
      chart(data,`probe_accuracy_unscaled_${attribute}`,'Label prediction / linear-probe accuracy',false,`probe_majority_${attribute}`,compact),
      chart(data,`probe_balanced_accuracy_unscaled_${attribute}`,'Label prediction / balanced accuracy',false,null,compact),
      chart(data,`retrieval_p10_${attribute}`,'Neighbour label agreement / P@10',false,`retrieval_chance_${attribute}`,compact)
    );
    document.querySelectorAll('.uncertainty-group').forEach(group => {
      group.hidden = group.dataset.attribute !== attribute;
    });
    document.getElementById('uncertainty-attribute').value = attribute;
  }
  fetch('data.json').then(response => {
    if (!response.ok) throw new Error('Missing endpoint data');
    return response.json();
  }).then(data => {
    render(data);
    document.getElementById('attribute').addEventListener('change', () => render(data));
    document.querySelector('.uncertainty-controls').hidden = false;
    document.getElementById('uncertainty-attribute').addEventListener('change', event => {
      document.getElementById('attribute').value = event.target.value;
      render(data);
    });
    const charts = document.getElementById('measured-charts');
    new ResizeObserver(() => {
      const columns = getComputedStyle(charts).gridTemplateColumns.split(' ').length;
      if (layout !== `${charts.clientWidth}:${columns}`) render(data);
    }).observe(charts);
  }).catch(() => {document.getElementById('data-error').hidden=false;});
})();

/* ------------------------------------------------------------------
   Animated architecture figure.

   The condition table is the same one in training/strategy.py. The phases
   trace Trainer.train_step: forward_pass, loss, backward, optimizer.step
   plus post_step_update. Nothing here encodes a measured value.
   ------------------------------------------------------------------ */
(() => {
  'use strict';
  const figure = document.getElementById('pipeline');
  if (!figure) return;
  // Reveals the controls. Without this script the figure stays the static
  // schematic and no inert buttons are shown.
  figure.dataset.interactive = 'true';

  const CONDITIONS = {
    ema_stopgrad:          { stopgrad:'on',  ema:'on',  sigreg:'off', projector:'off' },
    none_stopgrad:         { stopgrad:'on',  ema:'off', sigreg:'off', projector:'off' },
    sigreg_stopgrad:       { stopgrad:'on',  ema:'off', sigreg:'on',  projector:'off' },
    sigreg_nostopgrad:     { stopgrad:'off', ema:'off', sigreg:'on',  projector:'off' },
    proj_sigreg_stopgrad:  { stopgrad:'on',  ema:'off', sigreg:'on',  projector:'on'  },
    proj_sigreg_nostopgrad:{ stopgrad:'off', ema:'off', sigreg:'on',  projector:'on'  },
    none_nostopgrad:       { stopgrad:'off', ema:'off', sigreg:'off', projector:'off' }
  };

  const PHASES = ['forward', 'loss', 'backward', 'update'];
  const select = document.getElementById('pipeline-condition');
  const readout = document.getElementById('pipeline-readout');
  const targetNote = document.getElementById('target-note');
  const projectorLabel = document.getElementById('projector-label');
  const sigregLabel = document.getElementById('sigreg-label');
  const playButton = document.getElementById('pipeline-play');
  const phaseButtons = [...figure.querySelectorAll('.pipeline-phases button')];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const compact = window.matchMedia('(max-width: 700px)');

  function describe(phase, flags) {
    const reg = flags.projector === 'on' ? 'the projector output' : 'the pooled context embedding';
    switch (phase) {
      case 'forward':
        return ['01 Forward.', `Masked images reach the context encoder and predictor; full images reach the ${
          flags.ema === 'on' ? 'EMA target encoder' : 'shared encoder on the target side'}.`];
      case 'loss':
        return ['02 Loss.', `The predictor's output is compared with the masked target latents.${
          flags.sigreg === 'on' ? ` SIGReg acts on ${reg} in the same step.` : ' No regularizer is active in this condition.'}`];
      case 'backward':
        return ['03 Backward.', (flags.stopgrad === 'on'
          ? 'Gradient flows back through the predictor and the context encoder. The target latents are detached, so no gradient reaches the target branch.'
          : 'Gradient also flows through the target branch. Both branches contribute gradients to the shared encoder.') +
          (flags.sigreg === 'on' ? ` SIGReg contributes gradients through ${reg}.` : '')];
      case 'update':
        return ['04 Update.', `AdamW updates the context encoder${
          flags.projector === 'on' ? ', the predictor and the projector' : ' and the predictor'}.${
          flags.ema === 'on' ? ' The target encoder is then updated by exponential moving average; the optimizer never touches it.' : ' There is no separate target encoder to update.'}`];
    }
  }

  let phaseIndex = 0;
  let timer = null;
  let visible = false;
  let wantsPlayback = true;

  function apply() {
    const flags = CONDITIONS[select.value];
    Object.entries(flags).forEach(([key, value]) => { figure.dataset[key] = value; });
    const phase = PHASES[phaseIndex];
    figure.dataset.phase = phase;
    phaseButtons.forEach(button =>
      button.setAttribute('aria-pressed', String(button.dataset.phase === phase)));
    targetNote.textContent = flags.ema === 'on' ? 'EMA copy, frozen' : 'shared weights';
    projectorLabel.textContent = flags.projector === 'on' ? 'Projector' : 'Bypass (no projector)';
    sigregLabel.textContent = flags.sigreg === 'on' ? 'SIGReg' : 'SIGReg, not in this condition';
    const [label, text] = describe(phase, flags);
    readout.innerHTML = '';
    const strong = document.createElement('strong');
    strong.textContent = label;
    readout.append(strong, ' ' + text);
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    playButton.setAttribute('aria-pressed', 'false');
    playButton.textContent = 'Play step';
    figure.dataset.playing = 'false';
  }

  function start() {
    if (timer || !canPlay()) return;
    timer = setInterval(() => { phaseIndex = (phaseIndex + 1) % PHASES.length; apply(); }, 2600);
    playButton.setAttribute('aria-pressed', 'true');
    playButton.textContent = 'Pause';
    figure.dataset.playing = 'true';
  }

  phaseButtons.forEach(button => button.addEventListener('click', () => {
    wantsPlayback = false;
    stop();
    phaseIndex = PHASES.indexOf(button.dataset.phase);
    apply();
  }));
  select.addEventListener('change', apply);
  playButton.addEventListener('click', () => {
    wantsPlayback = !timer;
    if (wantsPlayback) start(); else stop();
  });

  // Only autoplay while the figure is visible and motion is welcome.
  const canPlay = () =>
    visible && !document.hidden &&
    !reduced.matches &&
    !compact.matches &&
    document.documentElement.classList.contains('motion-ready');

  function syncPlayback() {
    document.documentElement.classList.toggle('motion-ready', !reduced.matches && !compact.matches);
    if (wantsPlayback && canPlay()) start(); else stop();
  }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      visible = entries.some(entry => entry.isIntersecting);
      syncPlayback();
    }, { threshold: 0.35 }).observe(figure);
  }
  reduced.addEventListener('change', syncPlayback);
  compact.addEventListener('change', syncPlayback);
  document.addEventListener('visibilitychange', syncPlayback);

  apply();
  syncPlayback();
})();

/* Reading location and section navigation. Native anchors work without JavaScript. */
(() => {
  'use strict';
  const contents = document.querySelector('.contents');
  if (!contents) return;
  const links = [...contents.querySelectorAll('a[href^="#"]')];
  const entries = links.map(link => ({link, target:document.getElementById(link.hash.slice(1))}));
  const button = contents.querySelector('.contents-toggle');
  const current = contents.querySelector('.contents-current');
  const initialHash = location.hash;
  let alignInitialHash = Boolean(initialHash);
  let pending = false;
  let active;
  contents.dataset.interactive = 'true';
  function close() {
    contents.dataset.open = 'false';
    button.setAttribute('aria-expanded', 'false');
  }
  function locate() {
    pending = false;
    const position = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) + 20;
    let selected = entries[0];
    for (const entry of entries) {
      if (entry.target.getBoundingClientRect().top <= position) selected = entry;
    }
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) {
      // Closed disclosures are controls, not the content currently being read.
      selected = entries.slice().reverse().find(entry =>
        entry.target.tagName !== 'DETAILS' || entry.target.open
      );
    }
    if (selected === active) return;
    active = selected;
    links.forEach(link => link.removeAttribute('aria-current'));
    selected.link.setAttribute('aria-current', 'location');
    current.textContent = selected.link.textContent;
  }
  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(locate);
  }
  button.addEventListener('click', () => {
    const open = contents.dataset.open !== 'true';
    contents.dataset.open = String(open);
    button.setAttribute('aria-expanded', String(open));
  });
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', event => {
      const target = document.getElementById(link.hash.slice(1));
      const disclosure = target?.closest('details');
      if (disclosure && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) {
        event.preventDefault();
        alignInitialHash = false;
        disclosure.open = true;
        if (location.hash !== link.hash) history.pushState(null, '', link.hash);
        close();
        target.scrollIntoView({behavior:'auto'});
        disclosure.querySelector('summary').focus({preventScroll:true});
        schedule();
      }
    });
  });
  links.forEach(link => link.addEventListener('click', () => {
    close();
    if (innerWidth < 1200 && !document.getElementById(link.hash.slice(1))?.closest('details')) {
      button.focus({preventScroll:true});
    }
  }));
  contents.addEventListener('keydown', event => {
    if (event.key === 'Escape') { close(); button.focus(); }
  });
  window.addEventListener('scroll', schedule, {passive:true});
  window.addEventListener('resize', schedule);
  function revealHash() {
    if (location.hash !== initialHash) alignInitialHash = false;
    const target = document.getElementById(location.hash.slice(1));
    const disclosure = target?.closest('details');
    if (disclosure) disclosure.open = true;
    schedule();
  }
  window.addEventListener('hashchange', revealHash);
  // The measured figures load asynchronously and can move a deep-link target.
  const keepUserPosition = () => { alignInitialHash = false; };
  ['wheel', 'touchstart', 'keydown'].forEach(type => {
    window.addEventListener(type, keepUserPosition, {passive:true, once:true});
  });
  new ResizeObserver(() => {
    if (alignInitialHash && document.querySelector('.metric-chart')) {
      alignInitialHash = false;
      document.getElementById(initialHash.slice(1))?.scrollIntoView({behavior:'instant'});
    }
    schedule();
  }).observe(document.querySelector('main'));
  close();
  revealHash();
  locate();
})();
