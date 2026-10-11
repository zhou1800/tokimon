const readline = require('node:readline');
const vm = require('node:vm');

// Small DOM fixture for executing the production scripts across CDP connections.
class Element {
  constructor(tag, id, text = '', type = '') {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.innerText = text;
    this.textContent = text;
    this.type = type;
    this.isConnected = true;
    this.ownerDocument = document;
    this.value = '';
    this.files = [];
    this.clicks = 0;
  }
  getAttribute(name) { return this[name] || ''; }
  scrollIntoView() {}
  focus() {}
  dispatchEvent() {}
  click() { this.clicks += 1; }
}
class HTMLInputElement extends Element {
  constructor(id, type = 'text') { super('input', id, '', type); }
}
class HTMLAnchorElement extends Element {}
class Event {}

let document;
const location = {href: 'https://example.com/'};
const contexts = new Map();
const objects = new Map();
function reloadFixture() {
  document = {
    title: 'Reference fixture', body: {innerText: 'Continue'},
    querySelectorAll() { return this.elements; },
  };
  document.elements = [
    new Element('button', 'continue', 'Continue'),
    new HTMLInputElement('search'),
    new HTMLInputElement('upload', 'file'),
  ];
  for (const context of contexts.values()) context.document = document;
}
reloadFixture();
function contextFor(id) {
  if (!contexts.has(id)) {
    contexts.set(id, vm.createContext({
      document, location, Element, HTMLInputElement, HTMLAnchorElement,
      Event, InputEvent: Event, reloadFixture,
    }));
  }
  return contexts.get(id);
}

function call(method, params) {
  if (method === 'Page.getFrameTree') return {frameTree: {frame: {id: 'main'}}};
  if (method === 'Page.createIsolatedWorld') return {executionContextId: 2};
  if (method === 'Runtime.evaluate') {
    const value = vm.runInContext(params.expression, contextFor(params.contextId || 1));
    if (params.returnByValue === false) {
      const objectId = String(objects.size + 1);
      objects.set(objectId, value);
      return {result: {objectId}};
    }
    return {result: {value}};
  }
  if (method === 'DOM.setFileInputFiles') {
    objects.get(params.objectId).files = params.files.map(path => ({name: path.split('/').pop()}));
  }
  if (method === 'Runtime.callFunctionOn') {
    const fn = vm.runInContext('(' + params.functionDeclaration + ')', contextFor(2));
    return {result: {value: fn.call(objects.get(params.objectId))}};
  }
  if (method === 'Runtime.releaseObjectGroup') objects.clear();
  return {};
}

readline.createInterface({input: process.stdin}).on('line', line => {
  try {
    const {method, params} = JSON.parse(line);
    process.stdout.write(JSON.stringify(call(method, params)) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({exceptionDetails: {text: String(error)}}) + '\n');
  }
});
