const scriptURL = 'https://script.google.com/macros/s/AKfycbwW-LJNg_n_pPfyo5RiMOYDgQgjn8iiE-5CmVqqrWiFmxDLWsdIJZWnjqHsSORDpc3S/exec';

const form = document.forms['contact-form'];

form.addEventListener('submit', e => {
  e.preventDefault();

  // Select the submit button
  const submitButton = form.querySelector('input[type="submit"]');
    if (submitButton) {
      submitButton.disabled = true;
    }

  fetch(scriptURL, { method: 'POST', body: new FormData(form) })

    .then(() => { window.location.href = 'thankyou.html'; })

    .catch(error => {
      console.error('Error!', error.message);
      alert('There was an issue submitting the form. Please try again.');
    })
    .finally(() => {
      if (submitButton) {
        submitButton.disabled = false; // Re-enable the button if necessary
      }
    });
});