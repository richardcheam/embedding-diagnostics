"""Static presentation of accepted endpoints, never evaluation of embeddings.

Every measured mark carries its record identity and exact value. Conceptual
illustrations live in the HTML template and are explicitly distinguished from data.
"""

from html import escape
from math import log10
from statistics import mean


def text(x, y, value, anchor="start", css=""):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'class="{css}">{escape(str(value))}</text>')


def source(record, metric):
    value = record
    for key in metric.split('.'):
        value = value[key]
    prefix = ('training' if record['condition'] in ('ema_stopgrad', 'none_nostopgrad')
              else 'source')
    attrs = (f'data-{prefix}-condition="{record["condition"]}" '
             f'data-{prefix}-metric="{metric}" data-{prefix}-value="{value}"')
    return value, attrs


def figure(title, svg, caption, desc):
    return (f'<figure class="research-plot"><h3>{escape(title)}</h3>'
            f'<svg viewBox="{svg[0]}" role="img" aria-label="{escape(title)}">'
            f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>{svg[1]}</svg>'
            f'<figcaption>{caption}</figcaption></figure>')


def rows_plot(title, rows, metric, caption, maximum=100, ticks=(0, 25, 50, 75, 100),
              logarithmic=False, floor=None, bars=False):
    """Labels above each row keep the same composition readable on a phone."""
    top, gap, left, right = 60, 67, 24, 370
    bottom = top + (len(rows)-1)*gap + 25
    low = ticks[0] if logarithmic else 0
    def scale(v):
        return left + ((log10(v) if logarithmic else v)-low) / (maximum-low) * (right-left)
    parts = []
    for tick in ticks:
        x = scale(10**tick if logarithmic else tick)
        parts.append(f'<path d="M{x} 45V{bottom}" class="plot-gridline"/>')
        parts.append(text(x, bottom+26, f'10^{tick}' if logarithmic else tick, 'middle',
                          'plot-tick'))
    description = []
    if floor is not None:
        x = scale(floor)
        parts.append(f'<path d="M{x} 45V{bottom}" class="plot-floor"/>')
    for i, (record, label, pigment) in enumerate(rows):
        y = top + i*gap
        value, attrs = source(record, metric)
        # Percent metrics have a declared 0–100 scale; source data stay fractional.
        percent = metric.startswith(('attributes.', 'retrieval.')) or record.get('_percent', False)
        displayed = 100*value if percent else value
        x = scale(displayed)
        pretty = (f'{displayed:.2f}%' if percent else
                  f'{value:.3g}' if logarithmic else f'{value:.2f}')
        if metric == 'summary.identity_changed_queries':
            pretty = str(value)
        parts.extend([text(left, y-14, label, css='plot-label'),
                      text(right, y-14, pretty, 'end', 'plot-value'),
                      f'<path d="M{left} {y}H{right}" class="plot-track"/>'])
        if bars:
            parts.append(f'<rect x="{left}" y="{y-4}" width="{x-left}" height="8" '
                         f'fill="var(--{pigment})" {attrs}><title>'
                         f'{escape(label)}: {pretty}</title></rect>')
        else:
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="var(--{pigment})" '
                         f'{attrs}><title>{escape(label)}: {pretty}</title></circle>')
        description.append(f'{label}: {pretty}.')
    if floor is not None:
        caption += f' Dashed line: random-neighbour floor {floor:.2f}%.'
    return figure(title, (f'0 0 400 {bottom+45}', ''.join(parts)), caption,
                  ' '.join(description))


def compression_plot(data, attribute, endpoint, title):
    """Dimension uses a numeric axis; connecting lines only guide the eye."""
    records = {p['condition']: p for p in data['bdd'] if p['campaign'] == 'C2'}
    native = records['c2_native_768']
    metric = f'attributes.{attribute}.{endpoint}'
    def x(d):
        return 48+(d-128)/640*310

    reference, _ = source(native, metric)

    def y(v):
        return 52+(5-100*(v-reference))/15*180
    parts = []
    for tick in (-10, -5, 0, 5):
        py = y(reference+tick/100)
        css = 'paired-zero' if tick == 0 else 'plot-gridline'
        parts.extend([f'<path d="M48 {py}H358" class="{css}"/>',
                      text(37, py+5, f'{tick:+g}' if tick else '0', 'end', 'plot-tick')])
    parts.append(text(48, 25, 'Change from native / percentage points', css='plot-tick'))
    for dim in (128, 256, 512, 768):
        parts.append(text(x(dim), 258, dim, 'middle', 'plot-tick'))
    parts.append(text(200, 283, 'Embedding dimensions', 'middle', 'plot-tick'))
    descriptions = []
    for method, pigment in [('mrl', 'comparison'), ('pca', 'focus')]:
        selected = [records[f'c2_{method}_{d}'] for d in (128, 256, 512)] + [native]
        points = [(d, source(p, metric)[0]) for d, p in zip((128, 256, 512, 768), selected)]
        coordinates = ' '.join(f'{x(d)},{y(v)}' for d, v in points)
        dash = ' stroke-dasharray="5 4"' if method == 'pca' else ''
        parts.append(f'<polyline points="{coordinates}" fill="none" '
                     f'stroke="var(--{pigment})" stroke-width="2"{dash}/>')
        for (dim, value), record in zip(points[:-1], selected[:-1]):
            _, attrs = source(record, metric)
            mark = (f'<rect x="{x(dim)-5}" y="{y(value)-5}" width="10" height="10"'
                    if method == 'pca' else f'<circle cx="{x(dim)}" cy="{y(value)}" r="5"')
            parts.append(f'{mark} fill="var(--{pigment})" {attrs}><title>'
                         f'{method.upper()} {dim}: {100*value:.2f}%</title>'
                         f'</{"rect" if method == "pca" else "circle"}>')
            descriptions.append(f'{method.upper()} {dim}: {100*value:.2f}%.')
    value, attrs = source(native, metric)
    parts.append(f'<circle cx="{x(768)}" cy="{y(value)}" r="6" '
                 f'fill="var(--reference)" {attrs}><title>Native: {100*value:.2f}%</title>'
                 '</circle>')
    first = source(records['c2_mrl_128'], metric)[0]
    second = source(records['c2_pca_128'], metric)[0]
    caption = (f'Native {100*value:.2f}%. At 128 dimensions: MRL {100*first:.2f}%, '
               f'PCA {100*second:.2f}%.')
    return figure(title, ('0 0 400 298', ''.join(parts)), caption,
                  f'Native {100*value:.2f}%. ' + ' '.join(descriptions))


def coco_intervals(intervals):
    """Display the accepted grouped-bootstrap intervals, without recomputation."""
    left, right, bottom = 24, 370, 375
    def x(v):
        return left+(v+5)/7*(right-left)
    parts = []
    for tick in (-5, -2.5, 0, 2):
        px = x(tick)
        parts.extend([f'<path d="M{px} 45V{bottom}" '
                      f'class="{"paired-zero" if tick == 0 else "plot-gridline"}"/>',
                      text(px, bottom+27, f'{tick:+g}' if tick else '0', 'middle',
                           'plot-tick')])
    descriptions = []
    for i, (condition, direction) in enumerate([
        ('mrl_256', 't2i'), ('mrl_256', 'i2t'),
        ('mrl_128', 't2i'), ('mrl_128', 'i2t')]):
        p = {k: 100*v for k, v in intervals[condition][direction].items()}
        dim = condition.split('_')[1]
        label = f'{dim}d · {"Text → image" if direction == "t2i" else "Image → text"}'
        y = 86+i*87
        parts.append(text(left, y-19, label, css='plot-label'))
        parts.append(f'<g class="paired-ci" data-condition="{condition}" '
                     f'data-direction="{direction}" data-delta="{p["delta"]}" '
                     f'data-lower="{p["lower"]}" data-upper="{p["upper"]}">'
                     f'<path d="M{x(p["lower"])} {y}H{x(p["upper"])}" '
                     'class="paired-ci-line"/>'
                     f'<circle cx="{x(p["delta"])}" cy="{y}" r="5" '
                     'fill="var(--ink)"/></g>')
        value = f'{p["delta"]:+.2f} [{p["lower"]:+.2f}, {p["upper"]:+.2f}]'
        parts.append(text(right, y+27, value, 'end', 'plot-tick'))
        descriptions.append(f'{label}: {value} percentage points.')
    return figure('How uncertain are the Hit@10 changes?',
                  ('0 0 400 425', ''.join(parts)),
                  'Compressed minus native, in percentage points. Dots: paired change; '
                  'lines: conditional 95% bootstrap intervals. Zero is no change.',
                  ' '.join(descriptions))


def training_figures(data):
    rows = []
    for condition in ('ema_stopgrad', 'none_nostopgrad'):
        rows.append(({'condition': condition, **{
            k: mean(v) for k, v in data['conditions'][condition].items()}},
            'EMA reference' if condition == 'ema_stopgrad' else 'Unprotected control',
            'reference' if condition == 'ema_stopgrad' else 'control'))
    # The summary uses five-seed means. Full individual-seed plots remain in the appendix.
    plots = []
    for metric, title, maximum, ticks, log in [
        ('total_variance', 'Embedding spread', 3, (-5, -3, -1, 1, 3), True),
        ('probe_accuracy_unscaled_timeofday', 'Labels recovered by a classifier',
         100, (0, 25, 50, 75, 100), False),
        ('retrieval_p10_timeofday', 'Time-of-day agreement in search',
         100, (0, 25, 50, 75, 100), False)]:
        prepared = [(dict(p, _percent=not log), label, color) for p, label, color in rows]
        plots.append(rows_plot(title, prepared, metric,
                               'Five-seed means. ' + ('Logarithmic variance axis.' if log
                                                     else 'Percent; time-of-day labels.'),
                               maximum, ticks, logarithmic=log))
    return '<div class="plot-grid three training-summary">' + ''.join(plots) + '</div>'


def extension_figures(data, intervals):
    """Return the measured scenes from already-validated display records."""
    bdd = {p['condition']: p for p in data['bdd']}
    names = [('c1_pristine', 'Unchanged embeddings', 'reference'),
             ('c1_scale_contraction_0.99', 'Shrink magnitude', 'ink'),
             ('c1_mean_injection_0.99', 'Add a common offset', 'ink'),
             ('c1_rank_truncation_0.99', 'Keep eight directions', 'ink'),
             ('c1_isotropic_noise_0.99', 'Add severe noise', 'issue')]
    rows = [(bdd[c], name, pigment) for c, name, pigment in names]
    geometry = '<div class="plot-grid two">' + ''.join([
        rows_plot('RankMe rises under noise', rows, 'geometry.rankme',
                  'Raw singular-spectrum count. Higher means a broader spectrum, '
                  'not necessarily more useful information.', 768, (0, 192, 384, 576, 768)),
        rows_plot('Weather retrieval approaches chance', rows, 'attributes.weather.p10',
                  'P@10: fraction of ten neighbours sharing the weather label. '
                  'The same five conditions, on the same retained sample.',
                  floor=100*sum((p['val']/983)**2 for p in
                                bdd['c1_pristine']['attributes']['weather']['support'].values()))
    ]) + '</div>'
    contraction = bdd['c1_scale_contraction_0.99']
    pristine = bdd['c1_pristine']
    shrink = '<dl class="measurement-strip">'
    for label, metric, format_value in [
        ('Variance', 'geometry.total_variance', lambda v: f'{v:.6g}'),
        ('Weather balanced accuracy', 'attributes.weather.balanced_accuracy',
         lambda v: f'{100*v:.2f}%'),
        ('Weather retrieval P@10', 'attributes.weather.p10', lambda v: f'{100*v:.2f}%')]:
        a, _ = source(pristine, metric)
        b, _ = source(contraction, metric)
        shrink += (f'<div><dt>{label}</dt><dd><span>{format_value(a)}</span>'
                   f'<span aria-label="changes to">→</span><strong>{format_value(b)}</strong>'
                   '</dd></div>')
    shrink += '</dl>'
    compression = '<div class="plot-grid three">'
    for endpoint, measure in [('balanced_accuracy', 'Balanced accuracy'), ('p10', 'P@10')]:
        for attribute, name in [('weather', 'Weather'), ('scene', 'Scene'),
                                ('timeofday', 'Time of day')]:
            compression += compression_plot(data, attribute, endpoint, f'{name} / {measure}')
    compression += '</div>'
    precision_names = [
        ('c3_native_ref', 'Native reference', 'ink'),
        ('c3_native_fp16_gallery', 'FP16 · gallery only', 'reference'),
        ('c3_native_fp16_both', 'FP16 · queries + gallery', 'reference'),
        ('c3_native_int8_gallery', 'INT8 · gallery only', 'comparison'),
        ('c3_native_int8_both', 'INT8 · queries + gallery', 'comparison'),
        ('c3_native_fp32_arithmetic', 'FP32 · arithmetic only', 'ink'),
        ('c3_mean99_ref', 'Common-offset reference', 'ink'),
        ('c3_mean99_fp16_gallery', 'Offset + FP16 gallery', 'issue')]
    precision = [(bdd[c], label, pigment) for c, label, pigment in precision_names]
    numerics = '<div class="plot-grid two">' + ''.join([
        rows_plot('How many top-10 sets changed?', precision, 'summary.identity_changed_queries',
                  'Out of 983 queries. Native and common-offset rows each use their own '
                  'unperturbed reference. Zero changes are reported explicitly.',
                  200, (0, 50, 100, 150, 200), bars=True),
        rows_plot('How large was the score error?', precision, 'summary.max_score_error',
                  'Largest absolute cosine-score difference from the relevant reference. '
                  'Logarithmic axis; smaller error alone does not guarantee stable neighbours.',
                  -2, (-16, -12, -8, -4, -2), logarithmic=True)
    ]) + '</div>'
    coco = [(p, 'Native · 768 dimensions' if p['dimension'] == 768
             else f'Learned prefix · {p["dimension"]} dimensions', pigment)
            for p, pigment in zip(data['coco'], ('reference', 'focus', 'comparison'))]
    paired = '<div class="plot-grid three">' + ''.join(
        rows_plot(title, coco, f'retrieval.i2t.{metric}', caption, bars=True)
        for metric, title, caption in [
            ('hit@10', 'Find at least one caption', 'Image → text Hit@10. A query counts '
             'as a hit when any of its five recorded captions appears in the top ten.'),
            ('hit@1', 'Recover the first result', 'Image → text Hit@1. The highest-ranked '
             'result must be one of the five recorded captions.'),
            ('set_recall@10', 'Recover more of the five captions', 'Image → text caption '
             'recall@10. Fraction of the five recorded captions recovered in the top ten.')]
    ) + '</div>'
    extra = '<div class="plot-grid two">' + ''.join(
        rows_plot(title, coco, f'retrieval.{direction}.{metric}', caption, bars=True)
        for direction, metric, title, caption in [
            ('t2i', 'hit@10', 'Text → image / Hit@10',
             '5,000 caption queries, 1,000 image candidates; one recorded parent per caption.'),
            ('t2i', 'hit@1', 'Text → image / Hit@1',
             'Recorded parent recovered at the first rank.'),
            ('i2t', 'overlap@10', 'Image → text / neighbour overlap',
             'Percent of native top-10 identities retained. This is not relevance.'),
            ('t2i', 'overlap@10', 'Text → image / neighbour overlap',
             'Percent of native top-10 identities retained. This is not relevance.')]
    ) + '</div>'
    return {'__GEOMETRY_FIGURES__': geometry, '__CONTRACTION_STRIP__': shrink,
            '__COMPRESSION_FIGURES__': compression, '__NUMERICS_FIGURES__': numerics,
            '__PAIRED_FIGURES__': paired, '__PAIRED_EXTRA__': extra,
            '__COCO_INTERVALS__': coco_intervals(intervals)}
