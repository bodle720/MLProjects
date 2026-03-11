## AWS Ground Truth for Instance Segmentation

The following template defines a custom labeling task for AWS Ground Truth that can be used for
the CVDMS instance segmentation workflow. It enables annotators to easily label imagery containing
multiple instances across one or more classes.

In the example below, three classes are used: *person*, *animal*, and *inanimate*.

```html
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<crowd-form>
  <crowd-instance-segmentation
    name="annotatedResult"
    src="{{ task.input['source-ref'] | grant_read_access }}"
    header="Segment all instances of person, animal, and inanimate"
    labels="['person','animal','inanimate']"
  >
    <full-instructions header="Segmentation Instructions">
      <ol>
        <li>Inspect the image.</li>
        <li>Select a label and draw masks for every visible instance of that label.</li>
        <li>Repeat until all instances are labeled.</li>
      </ol>
    </full-instructions>

    <short-instructions>
      <p>Create a mask for each instance of person, animal, and inanimate.</p>
    </short-instructions>
  </crowd-instance-segmentation>
</crowd-form>

```