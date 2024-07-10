const inputFile = document.querySelector('#file');
const imgPreview = document.querySelector('#image-preview');

inputFile.addEventListener('change', function() {
    const file = this.files[0];
    if (file.size < 2000000) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imgPreview.src = e.target.result;
            imgPreview.style.display = 'block';
        }
        reader.readAsDataURL(file);
    } else {
        alert("Image size must be less than 2MB");
    }
});
