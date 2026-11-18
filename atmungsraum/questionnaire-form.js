import { likertLabels, mainQuestionnaireItems } from './data-logging.js?v=20261118-questionnaire-scales';

function measured(tag, text) {
  const element = document.createElement(tag);
  element.dataset.measure = '';
  element.textContent = text;
  return element;
}

export function mountQuestionnaireForm(form, status) {
  function changed(field) {
    field.firstControl.removeAttribute('aria-invalid');
    status.hidden = true;
  }

  const fields = mainQuestionnaireItems.map(({ type, prompt, scale }, index) => {
    const id = 'questionnaire-q' + (index + 1);
    const group = document.createElement('fieldset');
    group.className = 'questionnaire-item';
    group.id = id;
    const legend = measured('legend', prompt);
    legend.id = id + '-prompt';
    group.append(legend);
    const endpoints = document.createElement('div');
    endpoints.className = 'questionnaire-endpoints';
    endpoints.id = id + '-endpoints';
    endpoints.append(measured('span', scale.min_label), measured('span', scale.max_label));
    const field = { group };
    if (type === 'likert') {
      const options = document.createElement('div');
      options.className = 'questionnaire-options';
      const radios = likertLabels.map((label, offset) => {
        const target = document.createElement('label');
        const radio = document.createElement('input');
        radio.type = 'radio';
        radio.name = id;
        radio.value = String(offset + scale.min);
        radio.required = true;
        radio.setAttribute('aria-label', radio.value + ' – ' + label);
        radio.addEventListener('change', () => changed(field));
        target.append(radio, measured('span', radio.value));
        options.append(target);
        return radio;
      });
      field.firstControl = radios[0];
      field.read = () => {
        const selected = radios.find(radio => radio.checked);
        return selected ? Number(selected.value) : undefined;
      };
      group.append(options, endpoints);
    } else {
      const slider = document.createElement('input');
      const value = measured('span', '—');
      value.className = 'questionnaire-value';
      value.dataset.measureSample = String(scale.max);
      value.dataset.liveValue = '';
      value.setAttribute('aria-hidden', 'true');
      legend.after(value);
      slider.type = 'range';
      slider.name = id;
      slider.min = String(scale.min);
      slider.max = String(scale.max);
      slider.setAttribute('aria-labelledby', legend.id);
      slider.setAttribute('aria-describedby', endpoints.id);
      function setAnswered(answered) {
        slider.toggleAttribute('data-answered', answered);
        slider.setAttribute('aria-valuetext', answered ? slider.value : 'Noch keine Auswahl');
        value.textContent = answered ? slider.value : '—';
      }
      function select() {
        setAnswered(true);
        changed(field);
      }
      slider.addEventListener('input', select);
      // A tap can deliberately select the midpoint without changing the value.
      slider.addEventListener('click', select);
      slider.addEventListener('keydown', event => {
        if (event.key !== ' ' && event.key !== 'Enter') return;
        event.preventDefault();
        select();
      });
      field.firstControl = slider;
      field.reset = () => setAnswered(false);
      field.read = () => slider.hasAttribute('data-answered') ? slider.valueAsNumber : undefined;
      field.reset();
      group.append(slider, endpoints);
    }
    form.append(group);
    return field;
  });

  return {
    reset() {
      form.reset();
      for (const field of fields) {
        field.firstControl.removeAttribute('aria-invalid');
        field.reset?.();
      }
      status.hidden = true;
      status.textContent = '';
      form.scrollTop = 0;
    },
    read() {
      const values = fields.map(field => field.read());
      const missing = values.indexOf(undefined);
      if (missing === -1) return values;
      const { group, firstControl } = fields[missing];
      status.textContent = 'Bitte beantworte alle fünf Fragen.';
      status.hidden = false;
      firstControl.setAttribute('aria-invalid', 'true');
      group.scrollIntoView({ block: 'nearest' });
      firstControl.scrollIntoView({ block: 'nearest' });
      firstControl.focus({ preventScroll: true });
      return null;
    },
  };
}
