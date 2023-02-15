function import_images() {
    let image_names = []
    for (let i = 0; i < 20; i++) {
        for (let j = 0; j < 20; j++) {
            image_names.push(i.toString() + '-' + j.toString() + '.png')
        }
    }
    console.log(image_names)

    let images_elem = document.getElementById('images')
    for (let name of image_names) {
        let img = document.createElement('img')
        img.src = name
        img.id = name
        img.style.display = 'none'

        images_elem.appendChild(img)
    }
}
import_images()

let dist = 0
let pc = 0
let img = null

document.getElementById('pc').addEventListener("input", (event) => {
    change_pc(event)
});
document.getElementById('dist').addEventListener("input", (event) => {
    change_dist(event)
});

function change_pc(event) {
    pc = event.target.value
    change_image()
}
function change_dist(event) {
    dist = event.target.value
    change_image()
}

function change_image() {
    new_img = pc.toString() + '-' + dist.toString() + '.png'
    if (img == null){
        document.getElementById(new_img).style.display = 'block'
        img = new_img
        return
    }
    console.log(img)
    document.getElementById(img).style.display = 'none'
    document.getElementById(new_img).style.display = 'block'
    img = new_img
}

change_image()